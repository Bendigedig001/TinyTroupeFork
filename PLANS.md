# PLANS.md

## Objective
Deliver reliable, fast, and token-efficient TinyTroupe persona generation and image-question answering with strict compatibility.

## Milestones
- [x] M1: Correctness fixes in TinyPerson async multimodal flow.
- [x] M2: Factory stability fixes (failure path, sequential logging, async name uniqueness).
- [x] M3: OpenAI client cache-race hardening + JSON reminder idempotency.
- [x] M4: `main.py` reliability and prompt-cache retention control.
- [x] M5: Add and pass targeted tests for all modified behavior.
- [x] M6: Final verification, docs alignment, and acceptance report.

## File-by-File Tasks
- `tinytroupe/agent/tiny_person.py`
  - Extract shared stimuli payload -> user message builder.
  - Use shared builder in both `act()` and `act_async()`.
- `tinytroupe/factory/tiny_person_factory.py`
  - Initialize `sampled_characteristics` in sync generation.
  - Fix `_generate_people_sequentially()` error logging branch.
  - Harden async name uniqueness with global checks + atomic reservation.
  - Remove factory-level duplicate JSON reminder insertion.
- `tinytroupe/clients/openai_client.py`
  - Reconstruct cached dict entries via `_from_cached_format()` in race branch.
  - Mark race-branch responses as cached for cost stats.
  - Make JSON reminder insertion idempotent when explicit instruction exists.
- `main.py`
  - Remove top-level TinyTroupe image import.
  - Add lazy stdout-silenced image utility import.
  - Add `--prompt-cache-retention` (`in_memory`, `24h`, `off`) and apply in LLM mode.
- `tests/unit/*`
  - Add tests for async multimodal propagation, factory failure stability, cache race, JSON reminder idempotency, and `main.py` JSON-only stdout.
- `AGENTS.md`
  - Keep operational constraints and regression checklist current.

## Acceptance Criteria
- Image prompts are preserved as multimodal payloads in sync and async action paths.
- Persona generation failure paths are non-crashing and deterministic.
- Concurrent local cache usage does not produce dict/attribute response errors.
- `main.py` stdout is valid JSON without preamble contamination.
- Duplicate JSON reminder prompts are eliminated in effective request history.
- No breaking public API changes in TinyTroupe core interfaces.

## Risks and Mitigations
- Risk: async name reservation conflicts across concurrent runs.
  - Mitigation: lock-guarded global uniqueness checks before acceptance.
- Risk: response reconstruction mismatch for cached objects.
  - Mitigation: unit tests mocking `_to_cacheable_format`/`_from_cached_format` and concurrent calls.
- Risk: hidden stdout writes from import side effects.
  - Mitigation: lazy imports wrapped with stdout redirection and subprocess JSON parsing tests.
- Risk: over-constraining prompts may reduce generation quality.
  - Mitigation: preserve existing prompt semantics; only deduplicate reminders.

## Done Definition
- All milestone checkboxes completed.
- Targeted tests pass locally for modified files.
- No new lint/type/test regressions introduced by touched code.
- Root docs (`AGENTS.md`, `PLANS.md`) reflect final implemented behavior.
