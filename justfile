default: pre-commit

# run ruff
lint:
  ruff check src/

# run ruff --fix
lint-fix:
  ruff check src/ --fix

# run ruff format
format:
  ruff format src/

# run mypy
typecheck:
  mypy src/

alias pcm := pre-commit

# run pre-commit on all files
pre-commit:
  pre-commit run --all-files
