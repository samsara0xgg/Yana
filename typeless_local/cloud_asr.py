"""Recognition in the cloud while he talks: Alibaba's Qwen-Audio streaming model.

Audio goes up in 0.1 s frames from the moment the microphone opens, so when he
lets go only the last sentence is left to finish (0.69 s median on 100 of the
user's dictations, against 2.3 s for the same model on the whole clip). The
model returns punctuated, nearly verbatim text and takes the word list as hot
words.

Anything that goes wrong (no key, no network, a failed task, no answer in
time) returns None, and the caller hears the recording with Whisper instead.

The WebSocket client is the few parts of RFC 6455 this needs, on the standard
library, so the app bundle takes on no new package.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
import select
import socket
import ssl
import struct
import threading
import uuid
from urllib.parse import urlsplit

import numpy as np

from typeless_local.asr import Transcript

LOGGER = logging.getLogger(__name__)

KEY_ENV = "BAILIAN_API_KEY"
MODEL = "qwen-audio-3.1-asr-flash-streaming"
# The voice-input sibling: words come out one by one, and with polish on it
# drops fillers and false starts itself (11.9% vs 14.6% unpolished on the same
# 100 dictations, 0.62 s median after the stop).
MESSAGE_MODEL = "qwen-audio-3.1-asr-flash-message"
# Model Studio, international (Singapore): the key has to come from the same region.
URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
CONNECT_TIMEOUT_S = 5.0
# After the stop: how long the last sentence may take before Whisper hears the recording instead.
FINISH_TIMEOUT_S = 4.0
FRAME_SECONDS = 0.1
HOTWORD_WEIGHT = 4
MAX_HOTWORDS = 2000
# A sentence much quieter than his own voice is someone else: a song or a
# video playing nearby, which the streaming model writes down too.
BACKGROUND_RATIO = 0.25
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")


def available() -> bool:
    return bool(os.environ.get(KEY_ENV))


def hotwords(terms: list[str]) -> dict[str, int]:
    """The word list as the model's instant hot words, within its limits:
    at most 15 characters for a term with Chinese in it, 7 words for one without."""

    words: dict[str, int] = {}
    for term in terms:
        term = str(term or "").strip()
        if not term or term in words:
            continue
        if term.isascii():
            if len(term.split()) > 7:
                continue
        elif len(term) > 15:
            continue
        words[term] = HOTWORD_WEIGHT
        if len(words) >= MAX_HOTWORDS:
            break
    return words


class QwenStream:
    """One dictation: open on start, ``feed`` every microphone block, ``finish`` on stop."""

    def __init__(
        self,
        api_key: str,
        terms: list[str] | None = None,
        sample_rate: int = 16000,
        url: str = URL,
        model: str = MODEL,
        on_text=None,
    ) -> None:
        self.model = model
        # Called from the worker with all the text so far, as each word arrives.
        self._on_text = on_text
        self._api_key = api_key
        self._url = url
        self._sample_rate = sample_rate
        self._hotwords = hotwords(terms or [])
        self._frames: queue.Queue[bytes | None] = queue.Queue()
        self._pending = bytearray()
        self._frame_bytes = int(sample_rate * FRAME_SECONDS) * 2
        self._sentences: dict[int, dict] = {}
        self._finished = threading.Event()
        self._error = ""
        self._ws: _WebSocket | None = None
        self._closed = False
        self._cancelled = False

    def start(self) -> None:
        threading.Thread(target=self._run, name="qwen-asr", daemon=True).start()

    def feed(self, chunk: np.ndarray) -> None:
        """Audio thread: queue a block as 16-bit PCM. Cheap; never touches the network."""

        if self._closed:
            return
        samples = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
        self._pending += (samples * 32767).astype("<i2").tobytes()
        while len(self._pending) >= self._frame_bytes:
            self._frames.put(bytes(self._pending[: self._frame_bytes]))
            del self._pending[: self._frame_bytes]

    def finish(self, audio: np.ndarray, timeout: float = FINISH_TIMEOUT_S) -> Transcript | None:
        """After the stop: the text, or None when Whisper should hear it instead."""

        if not self._closed:
            self._closed = True
            if self._pending:
                self._frames.put(bytes(self._pending))
                self._pending.clear()
            self._frames.put(None)
        if not self._finished.wait(timeout):
            LOGGER.warning("Qwen ASR gave no final text within %.1fs; falling back to Whisper", timeout)
            self.cancel()
            return None
        if self._error:
            LOGGER.warning("Qwen ASR failed (%s); falling back to Whisper", self._error)
            return None
        sentences = [self._sentences[key] for key in sorted(self._sentences)]
        kept = drop_background(sentences, audio, self._sample_rate)
        text = "".join(str(sentence.get("text") or "") for sentence in kept).strip()
        language = "zh" if _CJK_RE.search(text) else "en"
        return Transcript(text=text, language=language, confidence=1.0)

    def cancel(self) -> None:
        """Stop: the worker leaves its loop, and closing the socket ends a read it is blocked in."""

        self._closed = self._cancelled = True
        self._frames.put(None)
        ws = self._ws
        if ws is not None:
            ws.close()

    # ------------------------------------------------------------ worker

    def _run(self) -> None:
        """Connect, start the task, then one loop sends queued frames and reads results.

        One thread does both: an SSL socket is not safe to read and write from two."""

        task_id = uuid.uuid4().hex
        try:
            ws = _WebSocket.connect(self._url, {"Authorization": f"Bearer {self._api_key}"}, CONNECT_TIMEOUT_S)
        except Exception as exc:
            self._fail(f"connect: {exc}")
            return
        self._ws = ws
        if self._cancelled:
            self._fail("cancelled")
            return
        parameters: dict = {"format": "pcm", "sample_rate": self._sample_rate}
        if self._hotwords:
            parameters["vocabulary"] = self._hotwords
        if self.model == MESSAGE_MODEL:
            parameters["disfluency_removal_enabled"] = True
            parameters["intermediate_result_enabled"] = self._on_text is not None
        try:
            ws.send_text(json.dumps({
                "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {
                    "task_group": "audio", "task": "asr", "function": "recognition",
                    "model": self.model, "parameters": parameters, "input": {},
                },
            }))
            if self._read_event(ws) != "task-started":
                self._fail("no task-started")
                return
            sending = True
            while not self._finished.is_set() and not self._cancelled:
                while sending:
                    try:
                        frame = self._frames.get_nowait()
                    except queue.Empty:
                        break
                    if frame is None:
                        sending = False
                        ws.send_text(json.dumps({
                            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                            "payload": {"input": {}},
                        }))
                    else:
                        ws.send_binary(frame)
                if ws.readable(0.02 if sending else 0.5) and self._read_event(ws) == "task-finished":
                    self._finished.set()
            ws.close()
        except Exception as exc:
            self._fail(str(exc))

    def _read_event(self, ws: _WebSocket) -> str:
        """Read one message; keep any sentence in it. Returns the event name."""

        message = json.loads(ws.recv_text())
        header = message.get("header") or {}
        event = str(header.get("event") or "")
        if event == "task-failed":
            raise RuntimeError(f"{header.get('error_code')}: {header.get('error_message')}")
        if event == "result-generated":
            sentence = ((message.get("payload") or {}).get("output") or {}).get("sentence") or {}
            if not sentence.get("heartbeat") and int(sentence.get("sentence_id") or 0) > 0:
                # A later result for the same sentence replaces the earlier one.
                self._sentences[int(sentence["sentence_id"])] = sentence
                self._tell_text()
        return event

    def _tell_text(self) -> None:
        if self._on_text is None or self._closed:
            return
        text = "".join(str(self._sentences[key].get("text") or "") for key in sorted(self._sentences))
        try:
            self._on_text(text)
        except Exception:
            LOGGER.exception("Showing the words so far failed")

    def _fail(self, why: str) -> None:
        if not self._finished.is_set():
            self._error = why
            self._finished.set()
        ws = self._ws
        if ws is not None:
            ws.close()


def drop_background(sentences: list[dict], audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Leave out sentences far quieter than the loudest one: speech or singing
    from a speaker in the room rather than his own voice into the microphone."""

    if len(sentences) < 2 or audio.size == 0:
        return sentences
    levels = []
    for sentence in sentences:
        begin = int(float(sentence.get("begin_time") or 0) / 1000 * sample_rate)
        end_ms = sentence.get("end_time")
        end = int(float(end_ms) / 1000 * sample_rate) if end_ms is not None else audio.size
        part = audio[max(0, begin) : max(begin + 1, min(end, audio.size))]
        levels.append(float(np.sqrt(np.mean(np.square(part, dtype=np.float64)))) if part.size else 0.0)
    loudest = max(levels)
    if loudest <= 0:
        return sentences
    kept = [s for s, level in zip(sentences, levels) if level >= loudest * BACKGROUND_RATIO]
    if len(kept) < len(sentences):
        LOGGER.info("Left out %d quiet sentence(s) as background sound", len(sentences) - len(kept))
    return kept


class _WebSocket:
    """A client WebSocket: text and binary out (masked), text in, pings answered."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._send_lock = threading.Lock()
        self._buffer = b""
        self._closed = False

    @classmethod
    def connect(cls, url: str, headers: dict[str, str], timeout: float) -> "_WebSocket":
        parts = urlsplit(url)
        secure = parts.scheme == "wss"
        port = parts.port or (443 if secure else 80)
        sock = socket.create_connection((parts.hostname, port), timeout=timeout)
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parts.hostname)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {parts.hostname}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            *(f"{name}: {value}" for name, value in headers.items()),
        ]
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        ws = cls(sock)
        response = ws._read_until(b"\r\n\r\n")
        status = response.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 101 " not in f"{status} ":
            sock.close()
            raise ConnectionError(f"handshake refused: {status}")
        sock.settimeout(None)
        return ws

    def readable(self, timeout: float) -> bool:
        """Whether a read would not block: bytes buffered here, in SSL, or on the socket."""

        if self._buffer:
            return True
        pending = getattr(self._sock, "pending", None)
        if pending is not None and pending():
            return True
        return bool(select.select([self._sock], [], [], timeout)[0])

    def send_text(self, text: str) -> None:
        self._send(0x1, text.encode())

    def send_binary(self, data: bytes) -> None:
        self._send(0x2, data)

    def recv_text(self) -> str:
        """The next text message; pings are answered and binary messages skipped on the way."""

        message, opcode = b"", None
        while True:
            fin, op, payload = self._read_frame()
            if op == 0x8:
                self.close()
                raise ConnectionError("closed by the server")
            if op == 0x9:
                self._send(0xA, payload)
                continue
            if op == 0xA:
                continue
            if op != 0x0:
                opcode, message = op, payload
            else:
                message += payload
            if fin:
                if opcode == 0x1:
                    return message.decode()
                message, opcode = b"", None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send(0x8, b"", force=True)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def _send(self, opcode: int, payload: bytes, force: bool = False) -> None:
        if self._closed and not force:
            raise ConnectionError("socket closed")
        header = bytes([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header += bytes([0x80 | length])
        elif length < 1 << 16:
            header += bytes([0x80 | 126]) + struct.pack("!H", length)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", length)
        mask = os.urandom(4)
        masked = (np.frombuffer(payload, dtype=np.uint8) ^ np.resize(np.frombuffer(mask, dtype=np.uint8), length)).tobytes() if length else b""
        with self._send_lock:
            self._sock.sendall(header + mask + masked)

    def _read_frame(self) -> tuple[bool, int, bytes]:
        first, second = self._read_exact(2)
        fin, opcode = bool(first & 0x80), first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if second & 0x80 else b""
        payload = self._read_exact(length)
        if mask:
            payload = (np.frombuffer(payload, dtype=np.uint8) ^ np.resize(np.frombuffer(mask, dtype=np.uint8), length)).tobytes()
        return fin, opcode, payload

    def _read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("connection closed")
            self._buffer += chunk
        data, self._buffer = self._buffer[:size], self._buffer[size:]
        return data

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self._buffer:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("connection closed during the handshake")
            self._buffer += chunk
        head, self._buffer = self._buffer.split(marker, 1)
        return head


def stream_if_set_up(terms: list[str], sample_rate: int, model: str = MODEL, on_text=None) -> QwenStream | None:
    """A started stream when the key is there; None leaves the dictation to Whisper."""

    key = os.environ.get(KEY_ENV, "")
    if not key:
        LOGGER.warning("Qwen recognition is on but %s is not set; using Whisper", KEY_ENV)
        return None
    stream = QwenStream(key, terms, sample_rate, model=model, on_text=on_text)
    stream.start()
    return stream
