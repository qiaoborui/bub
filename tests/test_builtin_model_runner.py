from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from any_llm.constants import LLMProvider
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletionChunk

from bub.builtin.model_runner import ModelRunner
from bub.builtin.settings import AgentSettings, ModelCandidate
from bub.builtin.tape import Tape
from bub.tape import AsyncTapeStoreAdapter, InMemoryTapeStore, TapeContext, TapeEntry


class _FakeStreamingOpenAIProvider(BaseOpenAIProvider):
    SUPPORTS_COMPLETION_STREAMING = True

    def __init__(self) -> None:
        self.completion_kwargs: dict[str, Any] | None = None

    async def acompletion(self, **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        self.completion_kwargs = kwargs
        include_usage = kwargs.get("stream_options") == {"include_usage": True}

        async def stream() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk.model_validate({
                "id": "chatcmpl_test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {"role": "assistant", "content": "done"},
                    }
                ],
            })
            final_chunk: dict[str, Any] = {
                "id": "chatcmpl_test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-test",
                "choices": [],
            }
            if include_usage:
                final_chunk["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            yield ChatCompletionChunk.model_validate(final_chunk)

        return stream()


class _FakeOpenAIModelRunner(ModelRunner):
    def __init__(self, settings: AgentSettings, llm: _FakeStreamingOpenAIProvider) -> None:
        super().__init__(settings)
        self._llm = llm

    def iter_llm_clients(self, model: str) -> Iterator[tuple[ModelCandidate, _FakeStreamingOpenAIProvider]]:
        yield ModelCandidate(provider=LLMProvider.OPENAI, model_id=model, name=f"openai:{model}"), self._llm


@pytest.mark.asyncio
async def test_streaming_openai_usage_is_requested_and_recorded_in_tape(tmp_path: Path) -> None:
    store = InMemoryTapeStore()
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(store), TapeContext()).scoped("test-tape")
    llm = _FakeStreamingOpenAIProvider()
    runner = _FakeOpenAIModelRunner(
        AgentSettings.model_construct(model="openai:gpt-test", max_tokens=100, model_timeout_seconds=None),
        llm,
    )

    await tape.ensure_bootstrap_anchor()
    events = [
        event async for event in runner.run(tape=tape, model="gpt-test", tools=[], system_prompt=None, prompt="hello")
    ]

    assert llm.completion_kwargs is not None
    assert llm.completion_kwargs["stream"] is True
    assert llm.completion_kwargs["stream_options"] == {"include_usage": True}
    assert [(event.kind, event.data) for event in events] == [
        ("text", {"delta": "done"}),
        ("final", {"ok": True, "text": "done"}),
    ]
    run_events = [
        entry for entry in store.read("test-tape") or [] if entry.kind == "event" and entry.payload.get("name") == "run"
    ]
    assert len(run_events) == 1
    assert run_events[0].payload["data"]["usage"] == {
        "completion_tokens": 2,
        "prompt_tokens": 3,
        "total_tokens": 5,
    }


@pytest.mark.asyncio
async def test_build_messages_omits_historical_images_for_text_model(tmp_path: Path) -> None:
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    await tape.ensure_bootstrap_anchor()
    await tape.store.append(
        tape.name,
        TapeEntry.message({
            "role": "user",
            "content": [
                {"type": "text", "text": "old image caption"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }),
    )
    runner = ModelRunner(
        AgentSettings.model_construct(
            model="deepseek:deepseek-v4-flash",
            vision_model="openai:gpt-5.4-mini",
        )
    )

    messages, _ = await runner.build_messages(
        tape=tape,
        run_id="run-1",
        system_prompt=None,
        prompt="follow up",
        model="deepseek:deepseek-v4-flash",
    )

    assert messages == [
        {"role": "user", "content": "old image caption"},
        {"role": "user", "content": "follow up"},
    ]


@pytest.mark.asyncio
async def test_build_messages_keeps_historical_images_for_vision_model(tmp_path: Path) -> None:
    image_content = [
        {"type": "text", "text": "old image caption"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    await tape.ensure_bootstrap_anchor()
    await tape.store.append(tape.name, TapeEntry.message({"role": "user", "content": image_content}))
    runner = ModelRunner(
        AgentSettings.model_construct(
            model="deepseek:deepseek-v4-flash",
            vision_model="openai:gpt-5.4-mini",
        )
    )

    messages, _ = await runner.build_messages(
        tape=tape,
        run_id="run-1",
        system_prompt=None,
        prompt="follow up",
        model="openai:gpt-5.4-mini",
    )

    assert messages[0]["content"] == image_content
