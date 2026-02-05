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
import contextlib
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional


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


def default_question_and_options() -> tuple[str, list[str]]:
    options = ["Tea", "Coffee", "Hot Chocolate"]
    question = "\n".join(
        [
            "Quick preference question:",
            "Rank the following options from most preferred (rank 1) to least preferred (rank 3).",
            f"Options: {', '.join(options)}",
            "",
            "Reply ONLY with valid JSON in this shape:",
            '{"ranking": ["Tea", "Coffee", "Hot Chocolate"], "why": "one short sentence"}',
            "",
            "Rules:",
            "- Use the exact option strings as provided (case/spelling).",
            "- Include each option exactly once in the ranking list.",
            "- No extra keys, no markdown, no surrounding text.",
        ]
    )
    return question, options


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _coerce_ranking(raw_ranking: Iterable[Any], options: list[str]) -> list[str]:
    canonical_by_norm = {_norm(opt): opt for opt in options}
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TinyTroupe UK personas + preference ranking + Borda scores (data only).")
    parser.add_argument("--people", type=int, default=3, help="Number of personas to generate (default: 3).")
    parser.add_argument("--mock", action="store_true", help="Run without any LLM calls (uses fixed sample personas).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for --mock mode.")
    args = parser.parse_args(argv)

    if args.people != 3:
        print("This demo is designed for exactly 3 personas (Borda with 3 options).", file=sys.stderr)
        print("Use `--people 3` (default).", file=sys.stderr)
        return 2

    question, options = default_question_and_options()
    use_mock = _should_use_mock(args.mock)

    if use_mock and not args.mock:
        print(
            "No API credentials detected; falling back to --mock mode. "
            "Set OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY) to run with the LLM.",
            file=sys.stderr,
        )

    responses: list[PersonResponse] = []

    if use_mock:
        for person in _mock_people(seed=args.seed)[: args.people]:
            # Deterministic preference => ranking
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
        TinyPersonFactory, TinyPerson, config_manager, utils = _silent_import_tinytroupe()

        # Keep output clean (JSON only).
        config_manager.update("loglevel", "ERROR")
        config_manager.update("loglevel_console", "ERROR")
        try:
            TinyPerson.communication_display = False  # type: ignore[attr-defined]
        except Exception:
            pass

        population = uk_population_sample()
        agent_particularities = "\n".join(
            [
                "UK resident; British English spelling/idioms.",
                "Generate a realistic, non-celebrity persona.",
                "Ensure the 3 generated people are clearly distinct in age, region, and occupation.",
            ]
        )

        factory = TinyPersonFactory.create_factory_from_demography(
            population,
            population_size=args.people,
            additional_demographic_specification="Target population: UK residents across England/Scotland/Wales/Northern Ireland.",
            context="You are participating in a short consumer preference survey in the United Kingdom.",
        )

        people: list[Any] = factory.generate_people(
            number_of_people=args.people,
            agent_particularities=agent_particularities,
            temperature=0.9,
            attempts=8,
            parallelize=False,
            verbose=False,
        )

        for person in people:
            actions = person.listen_and_act(
                question,
                return_actions=True,
                communication_display=False,
            )
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

            ranking = _coerce_ranking(raw_ranking, options)
            persona_summary = {
                k: person.get(k)
                for k in ["age", "gender", "nationality", "residence", "education", "occupation"]
                if person.get(k) is not None
            }
            responses.append(
                PersonResponse(
                    name=str(person.get("name") or getattr(person, "name", "")),
                    persona_summary=persona_summary,
                    raw_talk=raw_talk,
                    ranking=ranking,
                    borda_points=_borda_points_for_ranking(ranking),
                )
            )

    points_per_person = [r.borda_points for r in responses]
    totals = _borda_totals(points_per_person, options)
    max_total = max(totals.values()) if totals else 0
    winners = [opt for opt, score in totals.items() if score == max_total]

    output = {
        "population": uk_population_sample(),
        "question": question,
        "options": options,
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
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

