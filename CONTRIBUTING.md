# Contributing to MiniEval

Issues and pull requests are welcome.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/OmegaMc1331/minieval-bench.git
cd minieval-bench
uv sync
```

## Running the checks

Run the linter and the test suite before opening a pull request:

```bash
uv run ruff check .
uv run pytest
```

The test suite runs offline with the mock provider; no API keys are required. A quick end-to-end check:

```bash
uv run minieval-bench run --config benchmarks/example.yaml --output results/
```

## Code expectations

- Python 3.12+, standard library first.
- No explanatory comments: names and structure should carry the meaning.
- Type hints where they clarify signatures.
- Keep the surface small: no new dependency without discussion in an issue first.

## Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, ...

## Pull requests

- One logical change per PR, with a short description of the motivation.
- New behavior needs tests; bug fixes need a test that fails without the fix.
- Documentation (`README.md`, `CHANGELOG.md`) updated when user-facing behavior changes.

## Reporting bugs

Include the MiniEval version (`minieval-bench --version`), the command that failed, the error output, and the config or dataset trimmed to the minimum that reproduces the problem. Never include API keys.
