# Population JSON files (TinyPersonFactory) — Specification

This document describes the JSON format expected by TinyTroupe when you use a “population” file (like the ones in `examples/information/populations/*.json`) with `TinyPersonFactory.create_factory_from_demography(...)`.

## What TinyTroupe actually requires (strict)

There is **no hard-coded schema** for these JSON files.

In `TinyPersonFactory.create_factory_from_demography(...)`, TinyTroupe:
1. Loads the JSON file with Python’s `json.loads(...)`, and
2. Embeds the resulting Python value verbatim into the LLM prompt via `json.dumps(..., indent=4)`.

So the only strict requirements are:

- The file must contain **valid JSON**.
- The parsed JSON must be something that `json.dumps(...)` can serialize (i.e., standard JSON types: object/array/string/number/boolean/null).

Notes:
- The code path does **not** look for specific keys like `title` or `content`.
- A **top-level JSON object** is strongly recommended (and matches the examples), even though other JSON types would technically still be inserted into the prompt.

Relevant implementation: `tinytroupe/factory/tiny_person_factory.py` → `TinyPersonFactory.create_factory_from_demography()`.

## Recommended format (convention used by the example population files)

While not required, using the same shape as the bundled examples makes the data easier for both humans and the model to interpret.

### Top-level keys (recommended)

- `title` (string): Short name for the population.
- `description` (string): What the population represents and any scope notes.
- `source` (string or object): Provenance (dataset, report, model run, date, etc.).
- `content` (object): The actual demographic / cultural / behavioral information.

### Suggested `content` sections (examples, not mandatory)

Any keys are fine. Common sections in `examples/information/populations/*.json` include:

- `Demographics`
- `Personality Traits`
- `Political Affiliation`
- `Religion`
- `Occupations`
- `Interests and Hobbies`
- `Habits`
- `Public Opinions`

### Data patterns that work well

Because the JSON is used as prompt context (not programmatically parsed), the best format is the one that is **clear and easy to scan**:

- Prefer **explicit distributions** when possible (percentages, proportions, or counts).
  - Example: `"Age Distribution": { "0-14": "18.6%", "15-24": "13.1%", "25-54": "39.4%" }`
- Use **nested objects** for breakdowns (e.g., region → urban/rural → subregions).
- Use **arrays** for examples or enumerations.
  - Example: `"High": ["Region A", "Region B"]`
- Keep **units** explicit and consistent (`%`, `USD`, `years`, `millions`, etc.).
- Keep the JSON reasonably sized; extremely large JSON blobs can overwhelm the prompt and reduce quality.

## Token / size considerations (practical constraint)

The entire JSON content is inserted into an LLM prompt. Very large population files can:

- increase latency/cost,
- exceed the model’s context window,
- degrade sampling quality (important details get diluted).

If you find yourself adding dozens of pages of data, consider summarizing it into higher-level distributions and key segment descriptions.

## Templates

### Minimal (valid, but may produce weaker/less controlled sampling)

```json
{
  "title": "Example Population",
  "content": {
    "Demographics": {
      "Country": "Exampleland",
      "Age Distribution": {
        "18-34": "40%",
        "35-64": "50%",
        "65+": "10%"
      }
    }
  }
}
```

### Recommended starting template (matches bundled examples)

```json
{
  "title": "…",
  "description": "…",
  "source": "…",
  "content": {
    "Demographics": {},
    "Personality Traits": {},
    "Occupations": {},
    "Interests and Hobbies": {},
    "Habits": {},
    "Public Opinions": {}
  }
}
```

## Creating new population files (what “matters”)

- Focus on the **dimensions you want represented** in the generated agents (age, income, region, education, values, habits, etc.).
- Prefer **structured distributions** over long prose.
- If you have study-specific constraints that are not naturally “demographic” (e.g., “must include early adopters”, “must include people who hate spicy food”), you can usually express those via the `additional_demographic_specification` argument when calling `TinyPersonFactory.create_factory_from_demography(...)`, instead of forcing everything into the JSON.

