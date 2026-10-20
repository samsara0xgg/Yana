from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np

from typeless_local import cloud_asr


def _frame(opcode: int, payload: bytes) -> bytes:
    """A server frame: unmasked."""

    length = len(payload)
    if length < 126:
        return bytes([0x80 | opcode, length]) + payload
    if length < 1 << 16:
        return bytes([0x80 | opcode, 126]) + struct.pack("!H", length) + payload
    return bytes([0x80 | opcode, 127]) + struct.pack("!Q", length) + payload


class FakeServer:
    """Speaks just enough of the run-task protocol; ``script`` decides how it answers."""

    def __init__(self, script) -> None:
        self.script = script
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.url = f"ws://127.0.0.1:{self.listener.getsockname()[1]}/api-ws/v1/inference"
        self.run_task: dict = {}
        self.audio = b""
        self.headers = ""
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        conn, _ = self.listener.accept()
        self.conn = conn
        request = b""
        while b"\r\n\r\n" not in request:
            request += conn.recv(4096)
        self.headers = request.decode()
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
        self.buffer = b""
        self.script(self)

    def read(self) -> tuple[int, bytes]:
        def exact(n: int) -> bytes:
            while len(self.buffer) < n:
                chunk = self.conn.recv(65536)
                if not chunk:
                    raise ConnectionError
                self.buffer += chunk
            data, self.buffer = self.buffer[:n], self.buffer[n:]
            return data

        first, second = exact(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", exact(8))[0]
        mask = exact(4)
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(exact(length)))
        return first & 0x0F, payload

    def send(self, message: dict) -> None:
        self.conn.sendall(_frame(0x1, json.dumps(message).encode()))

    def event(self, name: str, **extra) -> None:
        self.send({"header": {"event": name, **extra}, "payload": {}})

    def sentence(self, sid: int, text: str, begin: int, end: int | None, final: bool = True) -> None:
        self.send({"header": {"event": "result-generated"}, "payload": {"output": {"sentence": {
            "sentence_id": sid, "text": text, "begin_time": begin, "end_time": end, "sentence_end": final}}}})


def _start_and_collect(server: FakeServer) -> None:
    _, payload = server.read()
    server.run_task = json.loads(payload)
    server.event("task-started")
    while True:
        opcode, payload = server.read()
        if opcode == 0x2:
            server.audio += payload
        elif opcode == 0x1 and json.loads(payload)["header"]["action"] == "finish-task":
            return


def _speech(seconds: float, level: float, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_streams_audio_and_returns_the_sentences():
    def script(server: FakeServer) -> None:
        _start_and_collect(server)
        server.sentence(1, "专门开个 session 做这个记忆层吧。", 0, 1000, final=False)
        server.sentence(1, "专门开个 session 做这个记忆层吧。", 0, 1000)
        server.sentence(2, "先定了。", 1000, 2000)
        server.event("task-finished")

    server = FakeServer(script)
    stream = cloud_asr.QwenStream("sk-test", ["言字", "Claude Code"], url=server.url)
    stream.start()
    audio = _speech(2.0, 0.3)
    for block in np.array_split(audio, 40):
        stream.feed(block)
    result = stream.finish(audio, timeout=5)

    assert result is not None
    assert result.text == "专门开个 session 做这个记忆层吧。先定了。"
    assert result.language == "zh"
    assert "Authorization: Bearer sk-test" in server.headers
    params = server.run_task["payload"]["parameters"]
    assert server.run_task["payload"]["model"] == cloud_asr.MODEL
    assert params["format"] == "pcm" and params["sample_rate"] == 16000
    assert params["vocabulary"] == {"言字": 4, "Claude Code": 4}
    assert len(server.audio) == audio.size * 2  # every sample, as 16-bit PCM


def test_a_quiet_sentence_in_the_background_is_left_out():
    def script(server: FakeServer) -> None:
        _start_and_collect(server)
        server.sentence(1, "这个先定了。", 0, 1000)
        server.sentence(2, "And I will always love you", 1000, 3000)
        server.event("task-finished")

    server = FakeServer(script)
    stream = cloud_asr.QwenStream("sk-test", url=server.url)
    stream.start()
    audio = np.concatenate([_speech(1.0, 0.4), _speech(2.0, 0.03)])
    stream.feed(audio)
    result = stream.finish(audio, timeout=5)

    assert result is not None
    assert result.text == "这个先定了。"


def test_a_failed_task_falls_back():
    def script(server: FakeServer) -> None:
        server.read()
        server.event("task-failed", error_code="InvalidApiKey", error_message="bad key")

    server = FakeServer(script)
    stream = cloud_asr.QwenStream("sk-test", url=server.url)
    stream.start()
    audio = _speech(0.5, 0.3)
    stream.feed(audio)
    assert stream.finish(audio, timeout=5) is None


def test_no_server_falls_back():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    stream = cloud_asr.QwenStream("sk-test", url=f"ws://127.0.0.1:{port}/")
    stream.start()
    audio = _speech(0.5, 0.3)
    stream.feed(audio)
    assert stream.finish(audio, timeout=5) is None


def test_no_answer_in_time_falls_back():
    def script(server: FakeServer) -> None:
        _start_and_collect(server)  # and then never finishes

    server = FakeServer(script)
    stream = cloud_asr.QwenStream("sk-test", url=server.url)
    stream.start()
    audio = _speech(0.5, 0.3)
    stream.feed(audio)
    assert stream.finish(audio, timeout=0.5) is None


def test_hotwords_keep_within_the_limits():
    words = cloud_asr.hotwords(["言字", "言字", "", "one two three four five six seven eight", "一" * 16, "Claude Code"])
    assert words == {"言字": 4, "Claude Code": 4}
