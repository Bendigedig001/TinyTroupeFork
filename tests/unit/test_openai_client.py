import asyncio
import threading
from types import SimpleNamespace

import pytest

from tinytroupe.clients.openai_client import OpenAIClient


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    to_dict=lambda: {"content": content, "role": "assistant"},
                )
            )
        ]
        self.usage = None


def _to_cached_dict(response: _FakeResponse) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": response.choices[0].message.content,
                    "role": "assistant",
                }
            }
        ]
    }


@pytest.mark.core
def test_json_reminder_is_not_duplicated_when_present():
    messages = [
        {
            "role": "system",
            "content": "Return a valid json object and nothing else.",
        },
        {"role": "user", "content": "produce output"},
    ]

    updated = OpenAIClient._ensure_json_keyword_in_messages(
        messages,
        response_format={"type": "json_object"},
    )

    reminders = [
        msg
        for msg in updated
        if isinstance(msg, dict)
        and msg.get("role") == "system"
        and "json object" in str(msg.get("content", "")).lower()
    ]

    assert len(reminders) == 1


@pytest.mark.core
def test_send_message_cache_race_reconstructs_existing_entry(tmp_path, monkeypatch):
    cache_file = tmp_path / "openai-client-cache-sync.pkl"
    client = OpenAIClient(cache_api_calls=True, cache_file_name=str(cache_file))

    client._concurrency_semaphore = None
    client._async_concurrency_semaphore = None

    monkeypatch.setattr(client, "_setup_from_config", lambda timeout=None: None)
    monkeypatch.setattr(client, "_save_cache", lambda: None)
    monkeypatch.setattr(client, "_count_tokens", lambda messages, model: 0)

    barrier = threading.Barrier(2)
    reconstructed_calls = {"count": 0}
    cost_flags: list[bool] = []

    def fake_raw_model_call(model, chat_api_params):
        barrier.wait(timeout=3)
        return _FakeResponse("live")

    def fake_extractor(response):
        assert hasattr(response, "choices")
        return {"content": response.choices[0].message.content}

    def fake_from_cached(cached_dict):
        reconstructed_calls["count"] += 1
        content = cached_dict["choices"][0]["message"]["content"]
        return _FakeResponse(content)

    def fake_update_cost_stats(response, was_cached):
        cost_flags.append(was_cached)

    monkeypatch.setattr(client, "_raw_model_call", fake_raw_model_call)
    monkeypatch.setattr(client, "_raw_model_response_extractor", fake_extractor)
    monkeypatch.setattr(client, "_to_cacheable_format", _to_cached_dict)
    monkeypatch.setattr(client, "_from_cached_format", fake_from_cached)
    monkeypatch.setattr(client, "_update_cost_stats", fake_update_cost_stats)

    messages = [{"role": "user", "content": "hello"}]
    results = []
    errors = []

    def worker():
        try:
            results.append(
                client.send_message(
                    messages,
                    model="gpt-4o-mini",
                    waiting_time=0,
                    max_attempts=1,
                    timeout=5,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors
    assert len(results) == 2
    assert all(result.get("content") == "live" for result in results)
    assert reconstructed_calls["count"] >= 1
    assert any(cost_flags)


@pytest.mark.core
def test_send_message_async_cache_race_reconstructs_existing_entry(tmp_path, monkeypatch):
    cache_file = tmp_path / "openai-client-cache-async.pkl"
    client = OpenAIClient(cache_api_calls=True, cache_file_name=str(cache_file))

    client._concurrency_semaphore = None
    client._async_concurrency_semaphore = None

    monkeypatch.setattr(client, "_save_cache", lambda: None)
    monkeypatch.setattr(client, "_count_tokens", lambda messages, model: 0)

    async def fake_setup_async_from_config(timeout=None):
        return None

    call_count = {"count": 0}
    gate = asyncio.Event()
    reconstructed_calls = {"count": 0}
    cost_flags: list[bool] = []

    async def fake_raw_model_call_async(model, chat_api_params):
        call_count["count"] += 1
        if call_count["count"] >= 2:
            gate.set()
        await gate.wait()
        return _FakeResponse("live-async")

    def fake_extractor(response):
        assert hasattr(response, "choices")
        return {"content": response.choices[0].message.content}

    def fake_from_cached(cached_dict):
        reconstructed_calls["count"] += 1
        content = cached_dict["choices"][0]["message"]["content"]
        return _FakeResponse(content)

    def fake_update_cost_stats(response, was_cached):
        cost_flags.append(was_cached)

    monkeypatch.setattr(client, "_setup_async_from_config", fake_setup_async_from_config)
    monkeypatch.setattr(client, "_raw_model_call_async", fake_raw_model_call_async)
    monkeypatch.setattr(client, "_raw_model_response_extractor", fake_extractor)
    monkeypatch.setattr(client, "_to_cacheable_format", _to_cached_dict)
    monkeypatch.setattr(client, "_from_cached_format", fake_from_cached)
    monkeypatch.setattr(client, "_update_cost_stats", fake_update_cost_stats)

    async def run_test():
        messages = [{"role": "user", "content": "hello"}]
        results = await asyncio.gather(
            client.send_message_async(
                messages,
                model="gpt-4o-mini",
                waiting_time=0,
                max_attempts=1,
                timeout=5,
            ),
            client.send_message_async(
                messages,
                model="gpt-4o-mini",
                waiting_time=0,
                max_attempts=1,
                timeout=5,
            ),
        )
        return results

    results = asyncio.run(run_test())

    assert len(results) == 2
    assert all(result.get("content") == "live-async" for result in results)
    assert reconstructed_calls["count"] >= 1
    assert any(cost_flags)
