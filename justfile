docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True *.py

lint: fmt
    ruff check *.py

fmt:
    isort .
    ruff format --preview .
