import asyncio
import copy
import sys

import pytest

# Insert paths at the beginning of sys.path (position 0)
sys.path.insert(0, "..")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../tinytroupe/")

from tinytroupe import control
from tinytroupe import config_manager
from tinytroupe.agent import TinyPerson
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.factory.tiny_factory import TinyFactory


@pytest.fixture(autouse=True)
def _reset_entities():
    control.reset()
    TinyPerson.clear_agents()
    TinyFactory.clear_factories()
    TinyPersonFactory.clear_factories()
    yield
    control.reset()
    TinyPerson.clear_agents()
    TinyFactory.clear_factories()
    TinyPersonFactory.clear_factories()


def _seed_stimulus(agent: TinyPerson, payload: dict) -> None:
    agent.store_in_memory(
        {
            "role": "user",
            "content": copy.deepcopy(payload),
            "type": "stimulus",
            "simulation_timestamp": agent.iso_datetime(),
        }
    )


@pytest.mark.core
def test_async_act_preserves_multimodal_stimulus_payload(monkeypatch):
    original_get = config_manager.get

    def fake_config_get(key, *args, **kwargs):
        if key == "api_type":
            return "openai"
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(config_manager, "get", fake_config_get)

    stimulus_payload = {
        "stimuli": [
            {
                "type": "VISUAL",
                "content": {
                    "text": "Please consider the attached image.",
                    "parts": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,ZmFrZQ=="
                            },
                        }
                    ],
                },
            }
        ]
    }

    sync_agent = TinyPerson("Sync Multimodal Tester")
    async_agent = TinyPerson("Async Multimodal Tester")

    _seed_stimulus(sync_agent, stimulus_payload)
    _seed_stimulus(async_agent, stimulus_payload)

    captured = {}

    def fake_generate_next_actions(person, messages):
        captured["sync"] = copy.deepcopy(messages[-1])
        return ([{"type": "DONE", "content": "", "target": ""}], "assistant", {}, [])

    async def fake_generate_next_actions_async(person, messages):
        captured["async"] = copy.deepcopy(messages[-1])
        return ([{"type": "DONE", "content": "", "target": ""}], "assistant", {}, [])

    monkeypatch.setattr(
        sync_agent.action_generator,
        "generate_next_actions",
        fake_generate_next_actions,
    )
    monkeypatch.setattr(
        async_agent.action_generator,
        "generate_next_actions_async",
        fake_generate_next_actions_async,
    )
    monkeypatch.setattr(sync_agent, "consolidate_episode_memories", lambda: None)

    async def fake_consolidate_episode_memories_async():
        return None

    monkeypatch.setattr(
        async_agent,
        "consolidate_episode_memories_async",
        fake_consolidate_episode_memories_async,
    )

    sync_agent.act(return_actions=True)
    asyncio.run(async_agent.act_async(return_actions=True))

    sync_message = captured["sync"]
    async_message = captured["async"]

    assert sync_message["role"] == "user"
    assert async_message["role"] == "user"
    assert isinstance(sync_message["content"], list)
    assert isinstance(async_message["content"], list)
    assert sync_message["content"] == async_message["content"]
    assert any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in async_message["content"]
    )
