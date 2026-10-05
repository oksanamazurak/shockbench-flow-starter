# Shortcuts for setting up and testing. Everything else is `uv run sbf <command>` (see `uv run sbf --help`).

install:
	uv sync
install_rl:
	uv sync --extra rl
install_evolve:
	uv sync --extra evolve
test:
	uv run pytest -n 3
lint:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: install install_rl install_evolve test lint
