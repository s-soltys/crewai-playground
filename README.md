# Crew Haiku

A simple CrewAI project that uses a local LLM (Ollama) to generate haikus about Python programming.

## Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Ollama installed with llama2 model

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Make sure Ollama is running with llama2 model:
```bash
ollama run llama2
```

## Usage

Run the project:
```bash
poetry run python main.py
```

## Configuration

The project configuration is stored in `config.toml`. You can modify:
- LLM settings (model, temperature)
- Agent properties
- Task description
