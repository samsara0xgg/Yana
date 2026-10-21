from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np
import pytest

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
    """Speaks just enough of the run-task protocol; ``script`` decides how it answers.

    Given several scripts it takes as many connections, one script each."""

    def __init__(self, *scripts) -> None:
        self.scripts = scripts
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.url = f"ws://127.0.0.1:{self.listener.getsockname()[1]}/api-ws/v1/inference"
        self.run_task: dict = {}
        self.audio = b""
        self.headers = ""
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        for script in self.scripts:
            conn, _ = self.listener.accept()
            self.conn = conn
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(4096)
            self.headers = request.decode()
            conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
            self.buffer = b""
            script(self)

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


def test_the_voice_input_model_polishes():
    def script(server: FakeServer) -> None:
        _, payload = server.read()
        server.run_task = json.loads(payload)
        server.event("task-started")
        while True:
            opcode, payload = server.read()
            if opcode == 0x1 and json.loads(payload)["header"]["action"] == "finish-task":
                break
        server.sentence(1, "专门开个 session。", 0, 1000)
        server.sentence(2, "先定了。", 1000, 2000)
        server.event("task-finished")

    server = FakeServer(script)
    stream = cloud_asr.QwenStream("sk-test", url=server.url, model=cloud_asr.MESSAGE_MODEL)
    stream.start()
    audio = _speech(2.0, 0.3)
    stream.feed(audio)
    result = stream.finish(audio, timeout=5)

    params = server.run_task["payload"]["parameters"]
    assert server.run_task["payload"]["model"] == "qwen-audio-3.1-asr-flash-message"
    assert params["disfluency_removal_enabled"] is True
    assert result is not None and result.text == "专门开个 session。先定了。"


@pytest.fixture(autouse=True)
def _extras_welcome(monkeypatch):
    """Each test starts as if the service had never refused the context or hints."""

    monkeypatch.setattr(cloud_asr, "_EXTRAS_REFUSED", threading.Event())


def test_hot_words_go_heaviest_first_and_never_above_five():
    words = cloud_asr.hotwords(["auto", "plain", "Jev", "Jarvis"], {"auto": 3, "Jev": 50, "Jarvis": 5})

    assert words == {"Jev": 5, "Jarvis": 5, "plain": 4, "auto": 3}
    assert list(words) == ["Jev", "Jarvis", "plain", "auto"]


def test_language_hints_only_when_they_cover_every_language_he_speaks():
    assert cloud_asr.language_hints(["zh", "en", "zh"]) == ["zh", "en"]
    assert cloud_asr.language_hints(["zh", "en", "fr", "de", "ja"]) == []  # one would be left out
    assert cloud_asr.language_hints(["zh", "yue"]) == []  # Cantonese is no hint the model takes
    assert cloud_asr.language_hints([]) == []


def test_context_keeps_the_last_five_texts_and_their_ends():
    turns = cloud_asr.context_messages(["", "old"] + ["x" * 390 + "末尾" * 10] + ["b", "c", "d", "e"])

    assert [turn["content"][0]["text"] for turn in turns][1:] == ["b", "c", "d", "e"]
    assert turns[0]["content"][0]["text"].endswith("末尾") and len(turns[0]["content"][0]["text"]) == 400
    assert all(turn["role"] == "user" and turn["content"][0]["type"] == "input_text" for turn in turns)


def test_a_letter_split_off_an_english_word_is_put_back():
    assert cloud_asr.tidy("我在用 agen t 做路由，agen t.") == "我在用 agent 做路由，agent."
    for kept in ("il y a un agent", "Plan B 先定", "see e.g. this", "用 C 语言", "3 x 4", "I have a dream", "okay, book it"):
        assert cloud_asr.tidy(kept) == kept


def test_ok_is_written_one_way_and_a_doubled_word_is_pulled_apart():
    assert cloud_asr.tidy("ok，Ok 好的 OK。") == "OK，OK 好的 OK。"
    assert cloud_asr.tidy("现在说话声音怎么样？okOK，我先用一段时间。") == "现在说话声音怎么样？OK OK，我先用一段时间。"
    assert cloud_asr.tidy("yesYes byebye TikTok Jev") == "yes Yes byebye TikTok Jev"


def test_sentences_cut_by_a_pause_keep_a_space_between_english_words():
    assert cloud_asr.join_sentences(["现在说话声音怎么样？", "ok", "OK", "我先用一段时间。", "Sounds good.", "Thanks"]) == (
        "现在说话声音怎么样？ok OK我先用一段时间。Sounds good. Thanks")


def test_the_task_starts_with_his_languages_and_the_context(caplog):
    caplog.set_level("INFO", logger=cloud_asr.LOGGER.name)
    server = FakeServer(_start_and_finish("Jev 的工具路由。"))
    stream = cloud_asr.QwenStream("sk-test", url=server.url, languages=["zh", "en"], context_pending=True)
    stream.start()
    stream.give_context(["上次说的 Jev", "Jev 工具路由 - Claude"])
    audio = _speech(1.0, 0.3)
    stream.feed(audio)
    result = stream.finish(audio, timeout=5)

    assert result is not None and result.text == "Jev 的工具路由。"
    assert server.run_task["payload"]["parameters"]["language_hints"] == ["zh", "en"]
    assert [turn["content"][0]["text"] for turn in server.run_task["payload"]["input"]["context"]] == [
        "上次说的 Jev", "Jev 工具路由 - Claude"]
    assert 'Qwen task: languages [\'zh\', \'en\'], 0 hot words, context ["上次说的 Jev", "Jev 工具路由 - Claude"]' in caplog.text


def test_context_that_never_comes_does_not_hold_the_task(monkeypatch):
    monkeypatch.setattr(cloud_asr, "CONTEXT_WAIT_S", 0.05)
    server = FakeServer(_start_and_finish("好的。"))
    stream = cloud_asr.QwenStream("sk-test", url=server.url, context_pending=True)
    stream.start()
    audio = _speech(1.0, 0.3)
    stream.feed(audio)

    result = stream.finish(audio, timeout=5)

    assert result is not None and result.text == "好的。"
    assert server.run_task["payload"]["input"] == {}


def test_a_refused_context_starts_again_without_it_and_is_left_out_after():
    def refuse(server: FakeServer) -> None:
        _, payload = server.read()
        server.refused = json.loads(payload)
        server.event("task-failed", error_code="InvalidParameter", error_message="context")

    server = FakeServer(refuse, _start_and_finish("好的。"))
    stream = cloud_asr.QwenStream("sk-test", url=server.url, languages=["zh"], context_pending=True)
    stream.start()
    stream.give_context(["窗口"])
    audio = _speech(1.0, 0.3)
    stream.feed(audio)
    result = stream.finish(audio, timeout=5)

    assert server.refused["payload"]["input"]["context"]
    assert result is not None and result.text == "好的。"
    assert server.run_task["payload"]["input"] == {} and "language_hints" not in server.run_task["payload"]["parameters"]
    assert len(server.audio) == audio.size * 2  # nothing lost while it started again
    assert cloud_asr._EXTRAS_REFUSED.is_set()


def _start_and_finish(text: str):
    def script(server: FakeServer) -> None:
        _start_and_collect(server)
        server.sentence(1, text, 0, 1000)
        server.event("task-finished")

    return script
