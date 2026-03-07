#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# One sync at a time (lock outside repo to keep working tree clean)
mkdir -p "${HOME}/.cache/pto-isa"
LOCK_FILE="${HOME}/.cache/pto-isa/sync_cann_to_github_main.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[$(date -Is)] Another sync is running; exiting." >&2
  exit 0
fi

cd "${REPO_DIR}"

# Safety checks
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "[$(date -Is)] ERROR: Working tree not clean. Aborting." >&2
  git status --porcelain=v1 >&2
  exit 2
fi

for r in cann origin; do
  if ! git remote get-url "${r}" >/dev/null 2>&1; then
    echo "[$(date -Is)] ERROR: remote '${r}' not found." >&2
    exit 3
  fi
done

echo "[$(date -Is)] Fetching remotes..."
git fetch cann --prune
git fetch origin --prune

if [[ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]]; then
  git checkout main
fi

# Ensure local main follows origin/main
if git show-ref --verify --quiet refs/remotes/origin/main; then
  echo "[$(date -Is)] Updating local main from origin/main (ff-only)..."
  git merge --ff-only origin/main || {
    echo "[$(date -Is)] ERROR: Cannot fast-forward main to origin/main. Aborting." >&2
    exit 4
  }
fi

if ! git show-ref --verify --quiet refs/remotes/cann/master; then
  echo "[$(date -Is)] ERROR: cann/master not found after fetch." >&2
  exit 5
fi

HEAD_SHA="$(git rev-parse HEAD)"
CANN_SHA="$(git rev-parse cann/master)"

if [[ "${HEAD_SHA}" == "${CANN_SHA}" ]]; then
  echo "[$(date -Is)] main already matches cann/master. Nothing to do."
  exit 0
fi

BASE="$(git merge-base HEAD cann/master || true)"
if [[ -n "${BASE}" && "${BASE}" == "${CANN_SHA}" ]]; then
  echo "[$(date -Is)] cann/master is behind main. Nothing to merge."
  exit 0
fi

MSG="sync: merge cann/master into main ($(date +%Y-%m-%d))"

echo "[$(date -Is)] Merging cann/master (${CANN_SHA}) into main (${HEAD_SHA})..."
if ! git merge --no-ff --no-edit -m "${MSG}" cann/master; then
  echo "[$(date -Is)] ERROR: Merge conflict. Aborting merge." >&2
  git merge --abort || true
  exit 6
fi

echo "[$(date -Is)] Pushing to origin main..."
git push origin main

echo "[$(date -Is)] DONE. main is now at $(git rev-parse HEAD)."
