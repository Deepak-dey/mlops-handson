#!/usr/bin/env bash
# Start the stack, then check service.py with black, isort, pylint and pytest.
#
#   ./run_checks.sh           check only (nothing is rewritten)
#   FIX=1 ./run_checks.sh     let black and isort reformat service.py first
#
# The tools come from the Pipfile:  pipenv install --dev black isort pylint pytest httpx

set -euo pipefail
cd "$(dirname "$0")/.."
export PIPENV_VERBOSITY=-1 

BLACK_FLAGS="--check"
ISORT_FLAGS="--check-only"
if [ "${FIX:-0}" = "1" ]; then
  BLACK_FLAGS=""
  ISORT_FLAGS=""
fi

[ -f .env ] || { echo "missing .env: run  cp .env.example .env  and fill it in"; exit 1; }

step() { printf '\n=== %s ===\n' "$1"; }

step "docker compose up"
make up

step "docker compose ps"
docker compose ps

step "black"
pipenv run black $BLACK_FLAGS service.py

step "isort"
pipenv run isort $ISORT_FLAGS service.py

step "pylint"
pipenv run pylint --fail-under=8 service.py

setp "pytest"
pipenv run pytest -v

printf '\nall steps passed\n'

