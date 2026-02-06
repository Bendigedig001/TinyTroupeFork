# AGENTS.md

## Mission
- Keep TinyTroupe persona factory + image-question flows reliable, token-efficient, and fast.
- Prioritize deterministic behavior, reproducible outputs, and machine-parseable interfaces.

## Guardrails
- Strict compatibility by default: no breaking API changes for public TinyTroupe classes/functions.
- No behavioral drift: preserve TinyTroupe process semantics (persona generation flow, action flow, memory flow).
- Prefer additive/internal changes and private helpers over contract changes.

## Caching Policy
- Use OpenAI prompt caching with stable prompt families/keys when available.
- Default retention for this repo pipeline: `in_memory` unless explicitly overridden.
- `off` means no `prompt_cache_retention` value should be sent.
- Avoid duplicate system reminders/instructions that reduce prompt-cache locality.
- Keep local disk cache keys deterministic and content-addressed.

## Async Policy
- Prefer native async end-to-end over thread bridges in async paths.
- Protect shared global name/caching state with explicit locking in concurrent paths.
- Never hold locks across network awaits.
- Bound concurrency using configured limits; avoid unbounded task fan-out.

## Test Gates (must pass before merge)
- Targeted unit tests for every bug fix and regression risk.
- JSON-only CLI output checks for `main.py` in mock and image-option paths.
- Cache-race safety tests under concurrent calls.
- Async multimodal payload propagation tests must match sync behavior.

## Regression Checklist
- Persona generation failure paths return `None` (no `UnboundLocalError`).
- Async + sync action paths preserve multimodal user content with images.
- OpenAI cache hit/miss behavior does not return raw dicts as model responses.
- JSON reminder insertion is idempotent when explicit instruction already exists.
- `main.py` emits stdout JSON without TinyTroupe startup/banner preamble.

## Rollback Rule
- If a change risks API/behavior compatibility, revert to last known-good behavior and reintroduce behind an opt-in flag.
- If partial rollout fails tests, disable the new path via config/flag defaults and keep old stable path active until fixed.
