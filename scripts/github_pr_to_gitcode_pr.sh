#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

GITHUB_EVENT_PATH="${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH is required}"
GITCODE_TOKEN="${GITCODE_TOKEN:?GITCODE_TOKEN is required}"
GITCODE_OWNER="${GITCODE_OWNER:-cann}"
GITCODE_REPO="${GITCODE_REPO:-pto-isa}"
GITCODE_BASE_BRANCH="${GITCODE_BASE_BRANCH:-master}"
GITCODE_HEAD_OWNER="${GITCODE_HEAD_OWNER:-zhhywang}"
GITCODE_REMOTE_NAME="${GITCODE_REMOTE_NAME:-gitcode}"
GITCODE_PUSH_URL="${GITCODE_PUSH_URL:?GITCODE_PUSH_URL is required}"
GITCODE_API_BASE="${GITCODE_API_BASE:-https://api.gitcode.com/api/v5}"
BRANCH_PREFIX="${BRANCH_PREFIX:-github-pr}"
DRY_RUN="${DRY_RUN:-0}"
GITCODE_PR_COMMENT="${GITCODE_PR_COMMENT:-/compile}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

json_get() {
  local expr="$1"
  jq -r "${expr}" "${GITHUB_EVENT_PATH}"
}

json_payload() {
  jq -n \
    --arg title "$1" \
    --arg head "$2" \
    --arg base "$3" \
    --arg body "$4" \
    '{"title": $title, "head": $head, "base": $base, "body": $body}'
}

print_pr_url() {
  local response_file="$1"
  local pr_url=""
  local pr_number=""

  pr_url="$(jq -r '.html_url // .web_url // .url // .links.html.href // empty' "${response_file}")"
  if [[ -n "${pr_url}" ]]; then
    log "GitCode PR URL: ${pr_url}"
    return 0
  fi

  pr_number="$(jq -r '.number // .iid // .id // empty' "${response_file}")"
  if [[ -n "${pr_number}" ]]; then
    log "GitCode PR URL: https://gitcode.com/${GITCODE_OWNER}/${GITCODE_REPO}/pulls/${pr_number}"
    return 0
  fi

  log "GitCode PR URL not found in API response"
}

command -v jq >/dev/null 2>&1 || die "jq is required"

pr_number="$(json_get '.pull_request.number')"
pr_action="$(json_get '.action')"
pr_title="$(json_get '.pull_request.title')"
pr_body="$(json_get '.pull_request.body // ""')"
pr_html_url="$(json_get '.pull_request.html_url')"
head_sha="$(json_get '.pull_request.head.sha')"
head_repo_full_name="$(json_get '.pull_request.head.repo.full_name')"
head_ref="$(json_get '.pull_request.head.ref')"
base_ref="$(json_get '.pull_request.base.ref')"

[[ -n "${pr_number}" ]] || die "cannot read pull request number"
[[ -n "${head_sha}" ]] || die "cannot read pull request head SHA"

# GitHub PR closed -> close the mirrored GitCode PR (matched by title) and stop.
# Best-effort: if no match is found, or the API call fails, just log and exit 0.
if [[ "${pr_action}" == "closed" ]]; then
  log "GitHub PR #${pr_number} closed; closing mirrored GitCode PR"
  list_url="${GITCODE_API_BASE}/repos/${GITCODE_OWNER}/${GITCODE_REPO}/pulls?state=open&per_page=100"
  gc_number="$(curl -sS --connect-timeout 30 --max-time 60 \
      -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" "${list_url}" \
    | jq -r --arg key "[GitHub PR #${pr_number}]" \
       '[.[] | select((.title // "") | contains($key)) | .number][0] // empty')"
  if [[ -z "${gc_number}" ]]; then
    log "no open GitCode PR matching '[GitHub PR #${pr_number}]'; nothing to close"
    exit 0
  fi
  close_url="${GITCODE_API_BASE}/repos/${GITCODE_OWNER}/${GITCODE_REPO}/pulls/${gc_number}"
  close_status="$(curl -sS --connect-timeout 30 --max-time 60 -o /dev/null -w '%{http_code}' \
    -X PATCH "${close_url}" \
    -H "Content-Type: application/json" \
    -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
    --data '{"state":"closed"}')"
  log "closed GitCode PR !${gc_number} (HTTP ${close_status})"
  exit 0
fi

gitcode_branch="${BRANCH_PREFIX}/${pr_number}"
gitcode_head="${GITCODE_HEAD_OWNER}:${gitcode_branch}"
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
  log "DRY_RUN=1; would rebase onto ${GITCODE_OWNER}/${GITCODE_REPO}:${GITCODE_BASE_BRANCH} and push to ${GITCODE_REMOTE_NAME}/${gitcode_branch}"
  log "DRY_RUN=1; would create GitCode PR '${gitcode_title}' from ${gitcode_head} to ${GITCODE_BASE_BRANCH}"
  exit 0
fi

# Rebase this PR's own commits onto cann:master so the GitCode PR (opened
# against cann:master) lists only these commits. Pushing the GitHub head
# verbatim would carry every commit the GitHub lineage has that cann:master
# lacks, dwarfing the actual change.
cann_git_url="https://oauth2:${GITCODE_TOKEN}@gitcode.com/${GITCODE_OWNER}/${GITCODE_REPO}.git"
mirror_base="refs/remotes/cann-base/${GITCODE_BASE_BRANCH}"
log "fetching ${GITCODE_OWNER}/${GITCODE_REPO}:${GITCODE_BASE_BRANCH} as the rebase base"
git fetch --no-tags "${cann_git_url}" "${GITCODE_BASE_BRANCH}:${mirror_base}"

mirror_branch="gitcode-mirror/${pr_number}"
git checkout -B "${mirror_branch}" "${mirror_base}"
# Ensure a commit identity exists (the runner may have none); used as committer
# for cherry-pick and as author of the snapshot commit below.
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Apply this PR's commits (base..head) onto cann:master. If they don't apply
# cleanly (the PR builds on GitHub-side changes cann:master lacks), fall back
# to a file snapshot: one squashed commit carrying the PR head's final version
# of every file the PR touches. The snapshot never conflicts, so the GitCode PR
# always shows the net file diff vs cann:master.
base_sha="$(json_get '.pull_request.base.sha')"
if [[ -n "${base_sha}" ]] && git rev-parse --verify --quiet "${base_sha}^{commit}" >/dev/null; then
  cherry_range="${base_sha}..${github_pr_ref}"
else
  # Base SHA not readable or unavailable: fall back to the head commit only.
  cherry_range="${github_pr_ref}"
fi
log "cherry-picking PR commit(s) ${cherry_range} onto ${GITCODE_BASE_BRANCH}"
if git cherry-pick "${cherry_range}"; then
  log "cherry-pick succeeded; PR commits applied cleanly"
else
  git cherry-pick --abort 2>/dev/null || true
  log "cherry-pick conflicted; falling back to file-snapshot (one squashed commit)"
  mapfile -t snapshot_files < <(git diff --name-only "${base_sha:-${github_pr_ref}^}" "${github_pr_ref}")
  [[ ${#snapshot_files[@]} -gt 0 ]] || die "no file changes detected for snapshot fallback"
  for f in "${snapshot_files[@]}"; do
    if git cat-file -e "${github_pr_ref}:$f" 2>/dev/null; then
      git checkout "${github_pr_ref}" -- "$f"
    else
      git rm -r --quiet -- "$f" 2>/dev/null || true
    fi
  done
  git add -A
  git commit --quiet -m "${gitcode_title}" \
    -m "Mirrored from GitHub PR #${pr_number}: ${pr_html_url} (snapshot: cherry-pick conflicted, net file diff vs ${GITCODE_BASE_BRANCH})"
fi

git push "${GITCODE_REMOTE_NAME}" "${mirror_branch}:refs/heads/${gitcode_branch}" --force

payload="$(json_payload "${gitcode_title}" "${gitcode_head}" "${GITCODE_BASE_BRANCH}" "${gitcode_body}")"
api_url="${GITCODE_API_BASE}/repos/${GITCODE_OWNER}/${GITCODE_REPO}/pulls"

log "creating GitCode PR from ${gitcode_head} to ${GITCODE_OWNER}/${GITCODE_REPO}:${GITCODE_BASE_BRANCH}"
response_file="$(mktemp)"
trap 'rm -f "${response_file}"' EXIT
status_code="$(
  curl -sS --connect-timeout 30 --max-time 180 \
    -o "${response_file}" -w '%{http_code}' \
    -X POST "${api_url}" \
    -H "Content-Type: application/json" \
    -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
    --data "${payload}"
)"

case "${status_code}" in
  200|201)
    log "GitCode PR created"
    print_pr_url "${response_file}"
    cat "${response_file}"
    # Best-effort: post a comment (default /compile) to trigger any bot-driven
    # CI on the new PR. Non-fatal — PR creation has already succeeded.
    if [[ -n "${GITCODE_PR_COMMENT}" ]]; then
      gc_pr_number="$(jq -r '.number // .iid // empty' "${response_file}")"
      if [[ -n "${gc_pr_number}" ]]; then
        comment_url="${GITCODE_API_BASE}/repos/${GITCODE_OWNER}/${GITCODE_REPO}/issues/${gc_pr_number}/comments"
        comment_status="$(curl -sS --connect-timeout 30 --max-time 60 -o /dev/null -w '%{http_code}' \
          -X POST "${comment_url}" \
          -H "Content-Type: application/json" \
          -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
          --data "$(jq -nc --arg b "${GITCODE_PR_COMMENT}" '{body:$b}')")"
        log "posted comment '${GITCODE_PR_COMMENT}' to GitCode PR !${gc_pr_number} (HTTP ${comment_status})"
      else
        log "could not parse GitCode PR number; skipping comment"
      fi
    fi
    ;;
  409|422)
    log "GitCode PR may already exist or request was rejected as duplicate"
    print_pr_url "${response_file}"
    cat "${response_file}"
    ;;
  *)
    cat "${response_file}" >&2
    die "GitCode API returned HTTP ${status_code}"
    ;;
esac
