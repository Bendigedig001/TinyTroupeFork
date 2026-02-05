# TinyTroupe

## Overview
TinyTroupe is an LLM-powered multiagent persona simulation library for imagination enhancement and business insights. It enables simulation of people with specific personalities, interests, and goals using Large Language Models (GPT-5 series).

## Project Structure
- `tinytroupe/` - Main Python package source code
  - `agent/` - TinyPerson agent implementation
  - `environment/` - TinyWorld simulation environments
  - `clients/` - LLM client implementations (OpenAI, Azure)
  - `utils/` - Utility functions and configuration
  - `ui/` - Jupyter widgets for interactive use
- `examples/` - Jupyter notebooks with example simulations
- `tests/` - Test suite
- `docs/` - Documentation and guides
- `data/` - Sample data and extractions

## Running the Project
The project runs as a Jupyter Lab server on port 5000.

**Workflow:** `Jupyter Notebook`
- Command: `JUPYTER_CONFIG_DIR=.jupyter jupyter lab --config=.jupyter/jupyter_notebook_config.py`
- Port: 5000

## Configuration
API configuration is in `config.ini`:
- Default model: `gpt-5-mini`
- Supports OpenAI and Azure OpenAI APIs

### Required Environment Variables
Set one of the following:
- `OPENAI_API_KEY` - For OpenAI API
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` - For Azure OpenAI

## Dependencies
Managed via `pyproject.toml`. Key dependencies:
- openai >= 1.65
- llama-index (with various embeddings)
- pandas, matplotlib, scipy
- jupyter, pydantic

## Development
- Python 3.11
- Tests: `pytest` (configured in pyproject.toml)
- Install locally: `pip install -e .`
