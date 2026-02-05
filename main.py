#!/usr/bin/env python3
"""
TinyTroupe demo script:
- Defines a small sample UK population JSON (embedded).
- Uses TinyPersonFactory to generate 3 personas (unless --mock).
- Asks each person a simple preference-ranking question.
- Prints responses + Borda scores as JSON (data only; no visuals).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import re
import hashlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from tinytroupe.utils.images import ImageSpec, build_image_content_part, preprocess_image_cached


def uk_population_sample() -> dict[str, Any]:
    # This is intentionally small and human-readable (prompt context, not parsed programmatically).
    return {
        "title": "United Kingdom — sample population (illustrative)",
        "description": (
            "A compact, synthetic demographic summary intended only for TinyTroupe demos. "
            "Not an authoritative statistical source."
        ),
        "source": {
            "type": "synthetic",
            "created_for": "TinyTroupe example script",
        },
        "content": {
            "Demographics": {
                "Country": "United Kingdom",
                "Nations (rough)": {
                    "England": "84%",
                    "Scotland": "8%",
                    "Wales": "5%",
                    "Northern Ireland": "3%",
                },
                "Age Distribution (rough)": {
                    "18-24": "9%",
                    "25-34": "14%",
                    "35-44": "13%",
                    "45-54": "13%",
                    "55-64": "12%",
                    "65+": "19%",
                },
                "Gender (rough)": {"Female": "51%", "Male": "49%"},
                "Urban/Rural (rough)": {"Urban": "83%", "Rural": "17%"},
            },
            "Education (rough)": {
                "No formal/low": "18%",
                "Secondary": "38%",
                "Further education/apprenticeship": "20%",
                "University degree+": "24%",
            },
            "Occupations (examples)": [
                "NHS and care work",
                "Retail and hospitality",
                "Construction and trades",
                "Education",
                "Finance and professional services",
                "Technology",
                "Transport and logistics",
            ],
            "Values & Lifestyle (examples)": {
                "Common concerns": [
                    "Cost of living",
                    "Housing affordability",
                    "Public services (NHS)",
                    "Transport",
                    "Work-life balance",
                ],
                "Leisure": [
                    "Pubs/cafés",
                    "Football and other sports",
                    "Streaming TV",
                    "Walking/hiking",
                    "Music and festivals",
                ],
            },
            "Language": {
                "Primary": "English (UK)",
                "Notes": [
                    "Use British spelling and idioms where natural.",
                    "Accents vary by region; don't overdo stereotypes.",
                ],
            },
        },
    }


@dataclass(frozen=True)
class ImageDefaults:
    max_dim: int = 768
    format: str = "jpeg"
    quality: int = 85
    detail: str = "low"


@dataclass(frozen=True)
class OptionSpec:
    id: str
    label: str
    images: list[ImageSpec]


def _default_option_specs() -> list[OptionSpec]:
    labels = ["Tea", "Coffee", "Hot Chocolate"]
    return [
        OptionSpec(id=chr(65 + i), label=label, images=[])
        for i, label in enumerate(labels)
    ]


def _build_question_text(
    options: list[OptionSpec],
    base_question: Optional[str] = None,
    include_images_info: bool = False,
) -> str:
    total = len(options)
    lines = [
        base_question or "Quick preference question:",
        f"Rank the following options from most preferred (rank 1) to least preferred (rank {total}).",
        "Options:",
    ]
    for opt in options:
        if include_images_info:
            lines.append(f"- {opt.id}: {opt.label} ({len(opt.images)} image(s))")
        else:
            lines.append(f"- {opt.id}: {opt.label}")

    if include_images_info:
        lines.append(
            "Images are attached below, grouped by option in the order listed; within each option they appear in the listed order."
        )

    lines.extend(
        [
            "",
            "Reply ONLY with valid JSON in this shape:",
            json.dumps(
                {"ranking": [opt.label for opt in options], "why": "one short sentence"},
                ensure_ascii=False,
            ),
            "",
            "Rules:",
            "- Use the exact option labels as provided above (case/spelling).",
            "- If you use option IDs instead of labels, use the IDs exactly as shown (e.g., A, B, C).",
            "- Include each option exactly once in the ranking list.",
            "- No extra keys, no markdown, no surrounding text.",
        ]
    )
    return "\n".join(lines)


def _load_options_spec(
    path: Optional[str],
    image_defaults: ImageDefaults,
) -> tuple[Optional[str], list[OptionSpec]]:
    if path is None:
        return None, _default_option_specs()

    payload = _read_json_file(path)
    question = payload.get("question")
    raw_options = payload.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("options JSON must include a non-empty 'options' list.")

    options: list[OptionSpec] = []
    for idx, opt in enumerate(raw_options):
        if isinstance(opt, str):
            label = opt
            opt_id = chr(65 + idx)
            images = []
        elif isinstance(opt, dict):
            label = opt.get("label") or opt.get("name") or opt.get("option")
            if not label:
                raise ValueError(f"Option at index {idx} missing label.")
            opt_id = opt.get("id") or chr(65 + idx)
            images = []
            raw_images = opt.get("images") or []
            if isinstance(raw_images, list):
                for img in raw_images:
                    if isinstance(img, str):
                        img_spec = {"path": img}
                    elif isinstance(img, dict):
                        img_spec = img
                    else:
                        continue
                    img_path = img_spec.get("path")
                    if not img_path:
                        continue
                    images.append(
                        ImageSpec(
                            path=img_path,
                            detail=img_spec.get("detail", image_defaults.detail),
                            max_dim=int(img_spec.get("max_dim", image_defaults.max_dim)),
                            format=img_spec.get("format", image_defaults.format),
                            quality=int(img_spec.get("quality", image_defaults.quality)),
                        )
                    )
        else:
            raise ValueError(f"Unsupported option value at index {idx}.")

        options.append(OptionSpec(id=str(opt_id), label=str(label), images=images))

    return question, options


def _build_question_payload(
    question_text: str,
    options: list[OptionSpec],
    cache_dir: str,
) -> tuple[Any, list[dict]]:
    has_images = any(opt.images for opt in options)
    image_fingerprints: list[dict] = []
    if not has_images:
        return question_text, image_fingerprints

    parts: list[dict] = []
    for opt in options:
        opt_images: list[dict] = []
        for img in opt.images:
            asset = preprocess_image_cached(img, cache_dir)
            opt_images.append(
                {
                    "source_sha256": asset.source_sha256,
                    "cache_key": asset.cache_key,
                    "format": asset.format,
                    "quality": asset.quality,
                    "max_dim": asset.max_dim,
                    "detail": img.detail,
                }
            )
            parts.append(build_image_content_part(asset, detail=img.detail))
        image_fingerprints.append(
            {"id": opt.id, "label": opt.label, "images": opt_images}
        )

    return {"text": question_text, "parts": parts}, image_fingerprints


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _write_json_file(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _coerce_ranking(
    raw_ranking: Iterable[Any], options: list[str], aliases: Optional[dict[str, str]] = None
) -> list[str]:
    canonical_by_norm = {_norm(opt): opt for opt in options}
    if aliases:
        for alias, target in aliases.items():
            if target in options:
                canonical_by_norm[_norm(alias)] = target
    ranking: list[str] = []
    for item in raw_ranking:
        if not isinstance(item, str):
            continue
        matched = canonical_by_norm.get(_norm(item))
        if matched and matched not in ranking:
            ranking.append(matched)
    for opt in options:
        if opt not in ranking:
            ranking.append(opt)
    return ranking[: len(options)]


def _borda_points_for_ranking(ranking: list[str]) -> dict[str, int]:
    n = len(ranking)
    return {option: (n - 1 - idx) for idx, option in enumerate(ranking)}


def _borda_totals(points_per_person: list[dict[str, int]], options: list[str]) -> dict[str, int]:
    totals = {opt: 0 for opt in options}
    for points in points_per_person:
        for opt in options:
            totals[opt] += int(points.get(opt, 0))
    return totals


@dataclass(frozen=True)
class PersonResponse:
    name: str
    persona_summary: dict[str, Any]
    raw_talk: str
    ranking: list[str]
    borda_points: dict[str, int]


def _silent_import_tinytroupe():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from tinytroupe.factory.tiny_person_factory import TinyPersonFactory  # type: ignore
        from tinytroupe.agent import TinyPerson  # type: ignore
        from tinytroupe import config_manager, utils  # type: ignore

    return TinyPersonFactory, TinyPerson, config_manager, utils


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


@contextlib.contextmanager
def _hard_timeout(label: str, timeout_s: Optional[float]):
    """
    Best-effort hard timeout for potentially stuck network calls.

    - On Unix (SIGALRM available) and in the main thread, we enforce a real timer.
    - Otherwise, we do nothing and rely on the client/library timeouts.
    """
    if timeout_s is None:
        yield
        return

    try:
        import signal
    except Exception:
        yield
        return

    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(_signum, _frame):
        raise TimeoutError(f"{label} timed out after {timeout_s:.0f}s")

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


def _should_use_mock(mock_flag: bool) -> bool:
    if mock_flag:
        return True
    if os.getenv("OPENAI_API_KEY"):
        return False
    if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return False
    return True


def _mock_people(seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    people = [
        {
            "name": "Priya Patel",
            "age": 29,
            "gender": "Female",
            "nationality": "British",
            "residence": "London, England, UK",
            "education": "University degree",
            "occupation": {"title": "Product Manager", "description": "Works at a mid-size fintech."},
            "preferences": {"likes": ["Coffee", "Podcasts"], "dislikes": ["Overly sweet drinks"]},
        },
        {
            "name": "Callum MacLeod",
            "age": 46,
            "gender": "Male",
            "nationality": "British",
            "residence": "Glasgow, Scotland, UK",
            "education": "Apprenticeship",
            "occupation": {"title": "Electrician", "description": "Self-employed tradesperson."},
            "preferences": {"likes": ["Tea", "Football"], "dislikes": ["Staying up late"]},
        },
        {
            "name": "Sharon Williams",
            "age": 21,
            "gender": "Female",
            "nationality": "British",
            "residence": "Manchester, England, UK",
            "education": "Further education",
            "occupation": {"title": "Retail Associate", "description": "Part-time while studying."},
            "preferences": {"likes": ["Hot Chocolate", "Streaming TV"], "dislikes": ["Bitter coffee"]},
        },
    ]
    rng.shuffle(people)
    return people


def _persona_cache_key(
    backend: str,
    mode: str,
    people: int,
    population: dict[str, Any],
    agent_particularities: str,
    model: Optional[str],
    temperature: float,
) -> str:
    payload = {
        "schema": "tinytroupe-uk-borda/personas/v1",
        "backend": backend,
        "mode": mode,
        "people": people,
        "population": population,
        "agent_particularities": agent_particularities,
        "model": model,
        "temperature": temperature,
    }
    return _sha256_text(_canonical_json(payload))


def _answers_cache_key(
    personas_key: str,
    question_text: str,
    options: list[str],
    image_fingerprints: list[dict],
) -> str:
    payload = {
        "schema": "tinytroupe-uk-borda/answers/v2",
        "personas_key": personas_key,
        "question": question_text,
        "options": options,
        "images": image_fingerprints,
    }
    return _sha256_text(_canonical_json(payload))


def _default_cache_dir() -> str:
    return os.path.join(os.getcwd(), ".tinytroupe_uk_borda_cache")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TinyTroupe UK personas + preference ranking + Borda scores (data only).")
    parser.add_argument("--people", type=int, default=3, help="Number of personas to generate (default: 3).")
    parser.add_argument(
        "--mode",
        choices=["direct", "demography"],
        default="direct",
        help="Generation mode: 'direct' (faster) or 'demography' (uses sampling plan).",
    )
    parser.add_argument("--mock", action="store_true", help="Run without any LLM calls (uses fixed sample personas).")
    parser.add_argument("--require-llm", action="store_true", help="Fail instead of falling back to --mock on LLM errors.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for --mock mode.")
    parser.add_argument("--model", type=str, default=None, help="Override TinyTroupe model (e.g., gpt-4.1-mini).")
    parser.add_argument("--timeout", type=float, default=30.0, help="Model call timeout in seconds (default: 30).")
    parser.add_argument(
        "--per-person-timeout",
        type=float,
        default=75.0,
        help="Hard timeout per persona generation (seconds, default: 75).",
    )
    parser.add_argument(
        "--per-question-timeout",
        type=float,
        default=60.0,
        help="Hard timeout per agent answer (seconds, default: 60).",
    )
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (default: 0.9).")
    parser.add_argument("--attempts", type=int, default=2, help="Attempts per persona generation (default: 2).")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=None,
        help="Parallelize independent LLM calls using true asyncio (persona generation + question answering). Enabled by default in LLM mode.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable async parallelism even in LLM mode.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max concurrent tasks when --parallel is set (default: TinyTroupe MAX_CONCURRENT_MODEL_CALLS).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=2048,
        help="Max completion tokens for model calls (default: 2048).",
    )
    parser.add_argument(
        "--options-json",
        type=str,
        default=None,
        help="Path to JSON file defining the question/options/images (optional).",
    )
    parser.add_argument(
        "--image-max-dim",
        type=int,
        default=768,
        help="Max image dimension (longest side) when preprocessing (default: 768).",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpeg",
        help="Image output format for preprocessing: jpeg, png, webp (default: jpeg).",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=85,
        help="Image quality for jpeg/webp preprocessing (default: 85).",
    )
    parser.add_argument(
        "--image-detail",
        type=str,
        default="low",
        help="OpenAI image detail parameter: low, high, auto (default: low).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=_default_cache_dir(),
        help="Directory for file persistence (default: ./.tinytroupe_uk_borda_cache).",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable file persistence (always regenerate personas/answers).",
    )
    parser.add_argument(
        "--refresh-personas",
        action="store_true",
        help="Ignore any cached personas and regenerate them.",
    )
    parser.add_argument(
        "--refresh-answers",
        action="store_true",
        help="Ignore any cached answers and re-ask the question.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages on stderr.")
    args = parser.parse_args(argv)

    if args.max_workers is not None and args.max_workers < 1:
        _eprint("--max-workers must be >= 1")
        return 2

    if args.people != 3:
        print("This demo is designed for exactly 3 personas (Borda with 3 options).", file=sys.stderr)
        print("Use `--people 3` (default).", file=sys.stderr)
        return 2

    image_defaults = ImageDefaults(
        max_dim=args.image_max_dim,
        format=args.image_format,
        quality=args.image_quality,
        detail=args.image_detail,
    )
    try:
        question_override, option_specs = _load_options_spec(args.options_json, image_defaults)
    except Exception as exc:
        _eprint(f"Failed to load options spec: {exc}")
        return 2

    if len(option_specs) != args.people:
        _eprint("Options count must match --people for this demo.")
        return 2

    include_images_info = any(opt.images for opt in option_specs)
    question_text = _build_question_text(
        option_specs, base_question=question_override, include_images_info=include_images_info
    )
    try:
        question_payload, image_fingerprints = _build_question_payload(
            question_text, option_specs, args.cache_dir
        )
    except Exception as exc:
        _eprint(f"Failed to preprocess images: {exc}")
        return 2
    options = [opt.label for opt in option_specs]
    option_aliases = {opt.id: opt.label for opt in option_specs}
    persist = (not args.no_persist)
    refresh_personas = bool(args.refresh_personas)
    refresh_answers = bool(args.refresh_answers) or refresh_personas
    use_mock = _should_use_mock(args.mock)
    backend = "mock" if use_mock else "llm"
    if backend == "llm":
        if args.no_parallel:
            args.parallel = False
        elif args.parallel is None:
            args.parallel = True

    if use_mock and not args.mock:
        _eprint(
            "No API credentials detected; falling back to --mock mode. "
            "Set OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY) to run with the LLM.",
        )

    responses: list[PersonResponse] = []

    population = uk_population_sample()
    agent_particularities = "\n".join(
        [
            "UK resident; British English spelling/idioms.",
            "Generate a realistic, non-celebrity persona.",
            "Ensure the 3 generated people are clearly distinct in age, region, and occupation.",
        ]
    )

    personas_key = _persona_cache_key(
        backend=backend,
        mode=args.mode,
        people=args.people,
        population=population,
        agent_particularities=agent_particularities,
        model=args.model,
        temperature=args.temperature,
    )
    personas_dir = os.path.join(args.cache_dir, "personas", personas_key)
    personas_meta_path = os.path.join(personas_dir, "meta.json")
    persona_paths = [os.path.join(personas_dir, f"person_{i+1}.agent.json") for i in range(args.people)]

    answers_key = _answers_cache_key(
        personas_key=personas_key,
        question_text=question_text,
        options=options,
        image_fingerprints=image_fingerprints,
    )
    answers_path = os.path.join(args.cache_dir, "answers", f"{answers_key}.json")

    # If answers exist, we can short-circuit the whole run (default behavior).
    if persist and (not refresh_answers) and os.path.exists(answers_path):
        try:
            cached = _read_json_file(answers_path)
            cached_responses = cached.get("responses")
            cached_options = cached.get("options")
            cached_question = cached.get("question")
            cached_images = cached.get("images", [])
            if (
                isinstance(cached_responses, list)
                and cached_options == options
                and cached_question == question_text
                and cached_images == image_fingerprints
                and len(cached_responses) == args.people
            ):
                responses = [
                    PersonResponse(
                        name=r.get("name", ""),
                        persona_summary=r.get("persona", {}) or {},
                        raw_talk=r.get("output", "") or "",
                        ranking=r.get("ranking", []) or [],
                        borda_points=r.get("borda_points", {}) or {},
                    )
                    for r in cached_responses
                    if isinstance(r, dict)
                ]
                if len(responses) == args.people:
                    if not args.quiet:
                        _eprint(f"Loaded cached answers from: {answers_path}")
            else:
                responses = []
        except Exception:
            responses = []

    # If no cached answers, ensure we have personas (load or generate), then ask and persist answers.
    if not responses:
        TinyPersonFactory = TinyPerson = config_manager = utils = None
        people_objects: list[Any] = []

        if not use_mock:
            TinyPersonFactory, TinyPerson, config_manager, utils = _silent_import_tinytroupe()

        try:
            if use_mock:
                if not args.quiet:
                    _eprint("Mode: mock (no LLM calls).")

                # Persist personas in the same format as TinyPerson spec files.
                if persist and (not refresh_personas) and all(os.path.exists(p) for p in persona_paths):
                    if not args.quiet:
                        _eprint(f"Found cached personas in: {personas_dir}")
                else:
                    os.makedirs(personas_dir, exist_ok=True)
                    people = _mock_people(seed=args.seed)[: args.people]
                    meta = {
                        "schema": "tinytroupe-uk-borda/personas/v1",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "backend": "mock",
                        "mode": args.mode,
                        "people": args.people,
                        "paths": [os.path.basename(p) for p in persona_paths],
                    }
                    _write_json_file(personas_meta_path, meta)
                    for i, person in enumerate(people):
                        spec = {"type": "TinyPerson", "persona": person}
                        _write_json_file(persona_paths[i], spec)
                    if not args.quiet and persist:
                        _eprint(f"Wrote personas to: {personas_dir}")

                # Compute deterministic "answers" without LLM.
                for spec_path in persona_paths:
                    spec = _read_json_file(spec_path)
                    person = spec.get("persona", {}) if isinstance(spec, dict) else {}
                    pref = (person.get("preferences") or {}).get("likes", [])
                    preferred = next((opt for opt in options if opt in pref), options[0])
                    remaining = [opt for opt in options if opt != preferred]
                    ranking = [preferred] + remaining
                    raw_talk = json.dumps({"ranking": ranking, "why": "Just a simple preference."})
                    responses.append(
                        PersonResponse(
                            name=str(person.get("name", "")),
                            persona_summary={k: person.get(k) for k in ["age", "gender", "nationality", "residence", "education", "occupation"] if person.get(k) is not None},
                            raw_talk=raw_talk,
                            ranking=ranking,
                            borda_points=_borda_points_for_ranking(ranking),
                        )
                    )
            else:
                # LLM mode: load cached personas if present (default), otherwise generate and persist.
                assert TinyPersonFactory is not None and TinyPerson is not None and config_manager is not None and utils is not None

                def _max_workers(task_count: int) -> int:
                    cap = args.max_workers
                    if cap is None:
                        cap = config_manager.get("max_concurrent_model_calls")

                    if cap is None:
                        return max(1, task_count)

                    try:
                        cap_int = int(cap)
                    except Exception:
                        cap_int = None

                    if cap_int is None or cap_int <= 0:
                        return max(1, task_count)

                    return max(1, min(task_count, cap_int))

                # Keep output clean (JSON only).
                config_manager.update("loglevel", "ERROR")
                config_manager.update("loglevel_console", "ERROR")
                config_manager.update("timeout", args.timeout)
                config_manager.update("max_completion_tokens", args.max_completion_tokens)
                config_manager.update("max_attempts", 2)
                if args.model:
                    config_manager.update("model", args.model)
                try:
                    TinyPerson.communication_display = False  # type: ignore[attr-defined]
                except Exception:
                    pass

                have_cached_personas = persist and (not refresh_personas) and all(os.path.exists(p) for p in persona_paths)
                if args.parallel:
                    # True-async end-to-end: one event loop for generation + questioning.
                    async def _run_llm_parallel() -> tuple[list[Any], list[PersonResponse]]:
                        people: list[Any] = []
                        try:
                            if have_cached_personas:
                                if not args.quiet:
                                    _eprint(f"Loading cached personas from: {personas_dir}")
                                for p in persona_paths:
                                    person_obj = TinyPerson.load_specification(
                                        p,
                                        suppress_mental_faculties=True,
                                        suppress_memory=True,
                                        suppress_mental_state=True,
                                    )
                                    people.append(person_obj)
                            else:
                                if not args.quiet:
                                    _eprint(
                                        f"Mode: {args.mode} (LLM). Generating {args.people} personas… "
                                        f"(timeout={args.timeout:.0f}s, per_person_timeout={args.per_person_timeout:.0f}s)"
                                    )

                                gen_start = time.perf_counter()
                                if args.mode == "demography":
                                    start = time.perf_counter()
                                    factory = TinyPersonFactory.create_factory_from_demography(
                                        population,
                                        population_size=args.people,
                                        additional_demographic_specification="Target population: UK residents across England/Scotland/Wales/Northern Ireland.",
                                        context="You are participating in a short consumer preference survey in the United Kingdom.",
                                    )
                                    if not args.quiet:
                                        _eprint(
                                            f"Factory created in {time.perf_counter() - start:.1f}s. Generating people…"
                                        )

                                    people = await asyncio.wait_for(
                                        factory.generate_people_async(
                                            number_of_people=args.people,
                                            agent_particularities=agent_particularities,
                                            temperature=args.temperature,
                                            attempts=args.attempts,
                                            parallelize=True,
                                            verbose=False,
                                            max_workers=_max_workers(args.people),
                                        ),
                                        timeout=args.per_person_timeout * max(1, args.people),
                                    )
                                else:
                                    factory = TinyPersonFactory(
                                        context=(
                                            "You are participating in a short consumer preference survey in the United Kingdom.\n\n"
                                            f"Population context (JSON):\n{json.dumps(population, indent=2)}"
                                        )
                                    )
                                    people = await asyncio.wait_for(
                                        factory.generate_people_async(
                                            number_of_people=args.people,
                                            agent_particularities=agent_particularities,
                                            temperature=args.temperature,
                                            attempts=args.attempts,
                                            parallelize=True,
                                            verbose=False,
                                            max_workers=_max_workers(args.people),
                                        ),
                                        timeout=args.per_person_timeout * max(1, args.people),
                                    )

                                if not people or len(people) < args.people:
                                    raise RuntimeError(
                                        f"Only generated {len(people)}/{args.people} personas."
                                    )

                                if not args.quiet:
                                    _eprint(
                                        f"Personas generated in {time.perf_counter() - gen_start:.1f}s."
                                    )

                                if persist:
                                    os.makedirs(personas_dir, exist_ok=True)
                                    meta = {
                                        "schema": "tinytroupe-uk-borda/personas/v1",
                                        "created_at": time.strftime(
                                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                        ),
                                        "backend": "llm",
                                        "mode": args.mode,
                                        "people": args.people,
                                        "model": args.model,
                                        "temperature": args.temperature,
                                        "paths": [
                                            os.path.basename(p) for p in persona_paths
                                        ],
                                    }
                                    _write_json_file(personas_meta_path, meta)
                                    for i, person_obj in enumerate(people):
                                        person_obj.save_specification(
                                            persona_paths[i],
                                            include_mental_faculties=False,
                                            include_memory=False,
                                            include_mental_state=False,
                                        )
                                    if not args.quiet:
                                        _eprint(f"Wrote personas to: {personas_dir}")

                            if not args.quiet:
                                _eprint("Asking preference question…")

                            ask_start = time.perf_counter()

                            semaphore = asyncio.Semaphore(_max_workers(len(people)))

                            async def _ask_and_parse_async(
                                person_obj: Any,
                            ) -> PersonResponse:
                                async with semaphore:
                                    actions = await asyncio.wait_for(
                                        person_obj.listen_and_act_async(
                                            question_payload,
                                            return_actions=True,
                                            communication_display=False,
                                        ),
                                        timeout=args.per_question_timeout,
                                    )

                                talk_actions = [
                                    a.get("action", {})
                                    for a in (actions or [])
                                    if isinstance(a, dict)
                                    and a.get("action", {}).get("type") == "TALK"
                                ]
                                raw_talk = (
                                    str(talk_actions[-1].get("content", ""))
                                    if talk_actions
                                    else ""
                                )

                                extracted = utils.extract_json(raw_talk) if raw_talk else {}
                                raw_ranking: Any = None
                                if isinstance(extracted, dict):
                                    if isinstance(extracted.get("ranking"), list):
                                        raw_ranking = extracted.get("ranking")
                                    elif isinstance(extracted.get("ranks"), dict):
                                        ranks = extracted["ranks"]
                                        raw_ranking = [
                                            k
                                            for k, _ in sorted(
                                                ranks.items(),
                                                key=lambda kv: float(kv[1]),
                                            )
                                        ]
                                elif isinstance(extracted, list):
                                    raw_ranking = extracted

                                if raw_ranking is None:
                                    raw_ranking = options

                                ranking = _coerce_ranking(raw_ranking, options, option_aliases)
                                persona_summary = {
                                    k: person_obj.get(k)
                                    for k in [
                                        "age",
                                        "gender",
                                        "nationality",
                                        "residence",
                                        "education",
                                        "occupation",
                                    ]
                                    if person_obj.get(k) is not None
                                }
                                resp = PersonResponse(
                                    name=str(
                                        person_obj.get("name")
                                        or getattr(person_obj, "name", "")
                                    ),
                                    persona_summary=persona_summary,
                                    raw_talk=raw_talk,
                                    ranking=ranking,
                                    borda_points=_borda_points_for_ranking(ranking),
                                )
                                if not args.quiet:
                                    _eprint(f"Ranking from {resp.name}: {resp.ranking}")
                                return resp

                            responses_local = await asyncio.gather(
                                *[_ask_and_parse_async(p) for p in people]
                            )

                            if not args.quiet:
                                _eprint(
                                    f"Answers collected in {time.perf_counter() - ask_start:.1f}s."
                                )

                            return people, list(responses_local)
                        finally:
                            try:
                                from tinytroupe.clients import client as tt_client  # type: ignore

                                c = tt_client()
                                closer = getattr(c, "aclose", None)
                                if closer is not None:
                                    result = closer()
                                    if asyncio.iscoroutine(result):
                                        await result
                            except Exception:
                                pass

                    people_objects, responses = asyncio.run(_run_llm_parallel())
                else:
                    if have_cached_personas:
                        if not args.quiet:
                            _eprint(f"Loading cached personas from: {personas_dir}")
                        for p in persona_paths:
                            person_obj = TinyPerson.load_specification(
                                p,
                                suppress_mental_faculties=True,
                                suppress_memory=True,
                                suppress_mental_state=True,
                            )
                            people_objects.append(person_obj)
                    else:
                        if not args.quiet:
                            _eprint(
                                f"Mode: {args.mode} (LLM). Generating {args.people} personas… "
                                f"(timeout={args.timeout:.0f}s, per_person_timeout={args.per_person_timeout:.0f}s)"
                            )

                        gen_start = time.perf_counter()
                        generated: list[Any] = []
                        if args.mode == "demography":
                            start = time.perf_counter()
                            factory = TinyPersonFactory.create_factory_from_demography(
                                population,
                                population_size=args.people,
                                additional_demographic_specification="Target population: UK residents across England/Scotland/Wales/Northern Ireland.",
                                context="You are participating in a short consumer preference survey in the United Kingdom.",
                            )
                            if not args.quiet:
                                _eprint(f"Factory created in {time.perf_counter() - start:.1f}s. Generating people…")

                            def _gen_people():
                                return factory.generate_people(
                                    number_of_people=args.people,
                                    agent_particularities=agent_particularities,
                                    temperature=args.temperature,
                                    attempts=args.attempts,
                                    parallelize=False,
                                    verbose=False,
                                )

                            with _hard_timeout(
                                label="generate_people",
                                timeout_s=args.per_person_timeout * max(1, args.people),
                            ):
                                generated = _gen_people()
                        else:
                            factory = TinyPersonFactory(
                                context=(
                                    "You are participating in a short consumer preference survey in the United Kingdom.\n\n"
                                    f"Population context (JSON):\n{json.dumps(population, indent=2)}"
                                )
                            )
                            for _ in range(args.people):
                                idx = len(generated) + 1
                                if not args.quiet:
                                    _eprint(f"Generating persona {idx}/{args.people}…")

                                def _gen_one():
                                    return factory.generate_person(
                                        agent_particularities=agent_particularities,
                                        temperature=args.temperature,
                                        attempts=args.attempts,
                                    )

                                with _hard_timeout(
                                    label=f"generate_person {idx}/{args.people}",
                                    timeout_s=args.per_person_timeout,
                                ):
                                    person_obj = _gen_one()
                                if person_obj is not None:
                                    generated.append(person_obj)

                        if not generated or len(generated) < args.people:
                            raise RuntimeError(f"Only generated {len(generated)}/{args.people} personas.")

                        if not args.quiet:
                            _eprint(f"Personas generated in {time.perf_counter() - gen_start:.1f}s.")

                        people_objects = list(generated)

                        if persist:
                            os.makedirs(personas_dir, exist_ok=True)
                            meta = {
                                "schema": "tinytroupe-uk-borda/personas/v1",
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "backend": "llm",
                                "mode": args.mode,
                                "people": args.people,
                                "model": args.model,
                                "temperature": args.temperature,
                                "paths": [os.path.basename(p) for p in persona_paths],
                            }
                            _write_json_file(personas_meta_path, meta)
                            for i, person_obj in enumerate(people_objects):
                                person_obj.save_specification(
                                    persona_paths[i],
                                    include_mental_faculties=False,
                                    include_memory=False,
                                    include_mental_state=False,
                                )
                            if not args.quiet:
                                _eprint(f"Wrote personas to: {personas_dir}")

                    # Ask question sequentially in sync mode.
                    if not args.quiet:
                        _eprint("Asking preference question…")

                    ask_start = time.perf_counter()

                    for person_obj in people_objects:
                        who = str(person_obj.get("name") or getattr(person_obj, "name", ""))
                        if not args.quiet:
                            _eprint(f"Asking: {who}…")

                        def _ask_one():
                            return person_obj.listen_and_act(
                                question_payload,
                                return_actions=True,
                                communication_display=False,
                            )

                        with _hard_timeout(
                            label=f"listen_and_act for {who}",
                            timeout_s=args.per_question_timeout,
                        ):
                            actions = _ask_one()

                        talk_actions = [
                            a.get("action", {})
                            for a in (actions or [])
                            if isinstance(a, dict) and a.get("action", {}).get("type") == "TALK"
                        ]
                        raw_talk = str(talk_actions[-1].get("content", "")) if talk_actions else ""

                        extracted = utils.extract_json(raw_talk) if raw_talk else {}
                        raw_ranking: Any = None
                        if isinstance(extracted, dict):
                            if isinstance(extracted.get("ranking"), list):
                                raw_ranking = extracted.get("ranking")
                            elif isinstance(extracted.get("ranks"), dict):
                                ranks = extracted["ranks"]
                                raw_ranking = [
                                    k for k, _ in sorted(ranks.items(), key=lambda kv: float(kv[1]))
                                ]
                        elif isinstance(extracted, list):
                            raw_ranking = extracted

                        if raw_ranking is None:
                            raw_ranking = options

                        ranking = _coerce_ranking(raw_ranking, options, option_aliases)
                        if not args.quiet:
                            _eprint(f"Ranking from {who}: {ranking}")
                        persona_summary = {
                            k: person_obj.get(k)
                            for k in ["age", "gender", "nationality", "residence", "education", "occupation"]
                            if person_obj.get(k) is not None
                        }
                        responses.append(
                            PersonResponse(
                                name=str(person_obj.get("name") or getattr(person_obj, "name", "")),
                                persona_summary=persona_summary,
                                raw_talk=raw_talk,
                                ranking=ranking,
                                borda_points=_borda_points_for_ranking(ranking),
                            )
                        )

                    if not args.quiet:
                        _eprint(f"Answers collected in {time.perf_counter() - ask_start:.1f}s.")
        except Exception as e:
            if args.require_llm:
                raise
            if not args.quiet:
                _eprint(f"LLM mode failed ({e}); falling back to mock personas for this run.")
            for person in _mock_people(seed=args.seed)[: args.people]:
                pref = (person.get("preferences") or {}).get("likes", [])
                preferred = next((opt for opt in options if opt in pref), options[0])
                remaining = [opt for opt in options if opt != preferred]
                ranking = [preferred] + remaining
                raw_talk = json.dumps({"ranking": ranking, "why": "Just a simple preference."})
                responses.append(
                    PersonResponse(
                        name=str(person.get("name", "")),
                        persona_summary={k: person.get(k) for k in ["age", "gender", "nationality", "residence", "education", "occupation"] if person.get(k) is not None},
                        raw_talk=raw_talk,
                        ranking=ranking,
                        borda_points=_borda_points_for_ranking(ranking),
                    )
                )

        if persist and responses:
            try:
                payload = {
                    "schema": "tinytroupe-uk-borda/answers/v2",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "personas_key": personas_key,
                    "question": question_text,
                    "options": options,
                    "images": image_fingerprints,
                    "responses": [
                        {
                            "name": r.name,
                            "persona": r.persona_summary,
                            "output": r.raw_talk,
                            "ranking": r.ranking,
                            "borda_points": r.borda_points,
                        }
                        for r in responses
                    ],
                }
                _write_json_file(answers_path, payload)
                if not args.quiet:
                    _eprint(f"Wrote answers to: {answers_path}")
            except Exception:
                pass

    points_per_person = [r.borda_points for r in responses]
    totals = _borda_totals(points_per_person, options)
    max_total = max(totals.values()) if totals else 0
    winners = [opt for opt, score in totals.items() if score == max_total]

    output = {
        "population": uk_population_sample(),
        "question": question_text,
        "options": options,
        "images": image_fingerprints,
        "responses": [
            {
                "name": r.name,
                "persona": r.persona_summary,
                "output": r.raw_talk,
                "ranking": r.ranking,
                "borda_points": r.borda_points,
            }
            for r in responses
        ],
        "borda_totals": totals,
        "borda_winner": winners[0] if len(winners) == 1 else winners,
        "persistence": {
            "enabled": persist,
            "cache_dir": args.cache_dir,
            "personas_key": personas_key,
            "answers_key": answers_key,
            "answers_path": answers_path if persist else None,
        },
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
