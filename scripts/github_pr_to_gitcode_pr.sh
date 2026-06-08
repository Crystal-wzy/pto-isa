#!/usr/bin/env bash
set -euo pipefail

GITHUB_EVENT_PATH="${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH is required}"
GITCODE_TOKEN="${GITCODE_TOKEN:?GITCODE_TOKEN is required}"
GITCODE_OWNER="${GITCODE_OWNER:-cann}"
GITCODE_REPO="${GITCODE_REPO:-pto-isa}"
GITCODE_BASE_BRANCH="${GITCODE_BASE_BRANCH:-master}"
GITCODE_REMOTE_NAME="${GITCODE_REMOTE_NAME:-gitcode}"
GITCODE_PUSH_URL="${GITCODE_PUSH_URL:?GITCODE_PUSH_URL is required}"
GITCODE_API_BASE="${GITCODE_API_BASE:-https://api.gitcode.com/api/v5}"
BRANCH_PREFIX="${BRANCH_PREFIX:-github-pr}"
DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

json_get() {
  local expr="$1"
  python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); value=eval(sys.argv[2], {}, {"data": data}); print("" if value is None else value)' \
    "${GITHUB_EVENT_PATH}" "${expr}"
}

json_payload() {
  local title="$1"
  local head="$2"
  local base="$3"
  local body="$4"
  python3 -c 'import json,sys; print(json.dumps({"title": sys.argv[1], "head": sys.argv[2], "base": sys.argv[3], "body": sys.argv[4]}, ensure_ascii=False))' \
    "${title}" "${head}" "${base}" "${body}"
}

urlencode() {
  python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
}

pr_number="$(json_get 'data["pull_request"]["number"]')"
pr_title="$(json_get 'data["pull_request"]["title"]')"
pr_body="$(json_get 'data["pull_request"].get("body")')"
pr_html_url="$(json_get 'data["pull_request"]["html_url"]')"
head_sha="$(json_get 'data["pull_request"]["head"]["sha"]')"
head_repo_full_name="$(json_get 'data["pull_request"]["head"]["repo"]["full_name"]')"
head_ref="$(json_get 'data["pull_request"]["head"]["ref"]')"
base_ref="$(json_get 'data["pull_request"]["base"]["ref"]')"

[[ -n "${pr_number}" ]] || die "cannot read pull request number"
[[ -n "${head_sha}" ]] || die "cannot read pull request head SHA"

gitcode_branch="${BRANCH_PREFIX}/${pr_number}"
gitcode_title="[GitHub PR #${pr_number}] ${pr_title}"
gitcode_body="$(cat <<EOF
Mirrored from GitHub PR #${pr_number}: ${pr_html_url}

GitHub source branch: ${head_repo_full_name}:${head_ref}
GitHub base branch: ${base_ref}
GitHub head SHA: ${head_sha}

---

${pr_body}
EOF
)"

github_pr_ref="refs/remotes/github-pr/${pr_number}"

log "mirroring GitHub PR #${pr_number} (${head_sha}) to GitCode branch ${gitcode_branch}"

if git remote get-url "${GITCODE_REMOTE_NAME}" >/dev/null 2>&1; then
  git remote set-url "${GITCODE_REMOTE_NAME}" "${GITCODE_PUSH_URL}"
else
  git remote add "${GITCODE_REMOTE_NAME}" "${GITCODE_PUSH_URL}"
fi

git fetch origin "pull/${pr_number}/head:${github_pr_ref}"
fetched_sha="$(git rev-parse "${github_pr_ref}")"
[[ "${fetched_sha}" == "${head_sha}" ]] || die "fetched PR ref ${fetched_sha} does not match event head ${head_sha}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1; would push ${github_pr_ref} to ${GITCODE_REMOTE_NAME}/${gitcode_branch}"
  log "DRY_RUN=1; would create GitCode PR '${gitcode_title}' against ${GITCODE_BASE_BRANCH}"
  exit 0
fi

git push "${GITCODE_REMOTE_NAME}" "${github_pr_ref}:refs/heads/${gitcode_branch}" --force

payload="$(json_payload "${gitcode_title}" "${gitcode_branch}" "${GITCODE_BASE_BRANCH}" "${gitcode_body}")"
encoded_token="$(urlencode "${GITCODE_TOKEN}")"
api_url="${GITCODE_API_BASE}/repos/${GITCODE_OWNER}/${GITCODE_REPO}/pulls?access_token=${encoded_token}"

log "creating GitCode PR against ${GITCODE_OWNER}/${GITCODE_REPO}:${GITCODE_BASE_BRANCH}"
response_file="$(mktemp)"
status_code="$(
  curl -sS -o "${response_file}" -w '%{http_code}' \
    -X POST "${api_url}" \
    -H "Content-Type: application/json" \
    --data "${payload}"
)"

case "${status_code}" in
  200|201)
    log "GitCode PR created"
    cat "${response_file}"
    ;;
  409|422)
    log "GitCode PR may already exist or request was rejected as duplicate"
    cat "${response_file}"
    ;;
  *)
    cat "${response_file}" >&2
    die "GitCode API returned HTTP ${status_code}"
    ;;
esac
