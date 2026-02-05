# Persona + TinyPersonFactory — Specification

This document describes the **persona** format used by `TinyPerson` and the **persona JSON** that `TinyPersonFactory` generates/consumes.

In TinyTroupe, a “persona” is just a JSON-serializable dictionary that gets embedded into the agent’s system prompt (see `tinytroupe/agent/prompts/tiny_person.v2.mustache`) and is used to drive realistic behavior.

## What TinyTroupe requires (strict)

### 1) If you load an agent spec file (`*.agent.json`)

`TinyPerson.load_specification(...)` expects a JSON object that at least contains:

- `type`: must be `"TinyPerson"`
- `persona`: an object (the persona dictionary)

Example skeleton:

```json
{
  "type": "TinyPerson",
  "persona": {
    "name": "Ada Example",
    "age": 30,
    "nationality": "Canadian",
    "occupation": "Accountant"
  }
}
```

Practical note: some helper methods (e.g., `TinyPerson.minibio()`) directly access keys like `age`, `nationality`, and `occupation`, so omitting them may cause runtime errors depending on what you call.

### 2) If you import a fragment (`*.fragment.json`)

`TinyPerson.import_fragment(path)` expects:

- `type`: must be `"Fragment"`
- `persona`: an object containing **partial** persona fields to merge into the agent

Example skeleton:

```json
{
  "type": "Fragment",
  "persona": {
    "preferences": {
      "likes": ["Long bike rides", "Podcasts"]
    }
  }
}
```

### 3) If you use `TinyPersonFactory`

The factory ultimately calls:

- `person = TinyPerson(agent_spec["name"])`
- `person.include_persona_definitions(agent_spec)`

So the only hard runtime requirement for the generated persona dict is:

- it must be valid JSON (object), and
- it must contain a **string** `name` field.

In practice, the factory prompt **requires** many more fields (next section), and providing them makes the simulation much higher quality.

## Persona schema used by `TinyPersonFactory` (expected)

`TinyPersonFactory`’s generator prompt (`tinytroupe/factory/prompts/generate_person.mustache`) instructs the model to output a JSON object with (at minimum) these fields:

- `name` (string)
- `age` (number)
- `gender` (string)
- `nationality` (string)
- `residence` (string)
- `education` (string)
- `long_term_goals` (array of strings)
- `occupation` (string or object; object is recommended)
- `style` (string)
- `personality` (object)
- `preferences` (object)
- `beliefs` (array of strings)
- `skills` (array of strings)
- `behaviors` (object)
- `health` (string or object)
- `relationships` (array or object; array of objects is common)
- `other_facts` (array of strings)

### Recommended shapes (conventions from bundled examples)

These are conventions used by `examples/agents/*.agent.json` (good starting point):

- `occupation` (object):
  - `title` (string)
  - `organization` (string, optional)
  - `description` (string; detailed)

- `personality` (object):
  - `traits` (array of strings)
  - `big_five` (object with keys like `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`)

- `preferences` (object):
  - `interests` (array of strings)
  - `likes` (array of strings)
  - `dislikes` (array of strings)

- `behaviors` (object):
  - `general` (array of strings)
  - `routines` (object with keys like `morning`, `workday`, `evening`, `weekend`; each an array of strings)

- `relationships` (array of objects), each like:
  - `name` (string)
  - `description` (string)

## How persona composition works (merging rules)

When you add persona definitions via `include_persona_definitions(...)` (used by both fragments and the factory), TinyTroupe merges dictionaries with these rules (`tinytroupe/utils/json.py:merge_dicts`):

- Missing keys are added.
- If the existing value is `null`/`None`, it is replaced.
- Dict + dict merges recursively.
- List + list concatenates and removes duplicates (preserving order).
- Scalar conflicts (two different non-null strings/numbers) raise an error by default.

If you need to **override** an existing scalar value, do it programmatically with `TinyPerson.define(...)` (overwrites scalars by default), instead of importing a conflicting fragment.

## Defining personas programmatically (no JSON files)

You can build a persona in code:

```python
from tinytroupe.agent import TinyPerson

person = TinyPerson("Ada Example")
person.define("age", 30)
person.define("nationality", "Canadian")
person.define("residence", "Toronto, Canada")
person.define("occupation", {"title": "Accountant", "description": "…"})
person.define("style", "Warm, concise, lightly humorous.")
```

Conveniences:

- Dictionary-style access: `person["nationality"] = "Canadian"`
- Add a dict or a fragment via `+`: `person + {"skills": ["…"]}` or `person + "path/to/fragment.json"`

## Constraining what a factory generates (agent particularities)

When calling `TinyPersonFactory.generate_person(...)` / `generate_people(...)`, `agent_particularities` is a **free-form string** that is inserted into the generator prompt.

The generator prompt explicitly supports constraints expressed as:

- concrete values (e.g., `occupation: "Nurse"`)
- numeric ranges (e.g., `age: [25, 40]`)
- lists (e.g., `gender: ["Male", "Female"]`)
- weighted choices (e.g., `financial_situation: {"rich": 0.1, "middle": 0.7, "poor": 0.2}`)
- or plain descriptive text (e.g., “Meticulous and competent, but not very nice.”)

If you already have a full persona dict you want to use, you typically **don’t need the factory**—load it as an `.agent.json` spec (or define it programmatically) instead.

