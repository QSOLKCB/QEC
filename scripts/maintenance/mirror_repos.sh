#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
ORG_TEST="${ORG_TEST:-EmergentMonk}"
ORG_DEV="${ORG_DEV:-multimodalas}"
DEST_ORG="${DEST_ORG:-QSOLKCB}"
TMPDIR="${TMPDIR:-$HOME/repo_mirror}"
DRY_RUN="${DRY_RUN:-0}"         # set to 1 to test without pushing
USE_HTTPS="${USE_HTTPS:-0}"     # set to 1 in CI; locally prefer SSH

# ======== Helpers =========
timestamp(){ date +"%F %T"; }
log(){ echo "[$(timestamp)] $*"; }
fail(){ echo "ERROR: $*" >&2; }

need(){ command -v "$1" >/dev/null || { fail "missing dependency: $1"; exit 1; }; }

need gh
need git

mkdir -p "$TMPDIR"

clone_repo() {
  local org="$1" repo="$2" dest="$TMPDIR/$org/$repo"
  mkdir -p "$TMPDIR/$org"
  if [[ -d "$dest/.git" ]]; then
    (cd "$dest" && git fetch --all --prune)
  else
    gh repo clone "$org/$repo" "$dest" -- --filter=blob:none
  fi
  echo "$dest"
}

ensure_branch() {
  local path="$1" branch="$2"
  ( cd "$path"
    git fetch origin --prune
    if git show-ref --verify --quiet "refs/heads/$branch"; then
      git checkout "$branch"
    elif git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
      git checkout -B "$branch" "origin/$branch"
    else
      local def
      def="$(git symbolic-ref --short refs/remotes/origin/HEAD | cut -d/ -f2)"
      git checkout -B "$branch" "origin/$def"
    fi
    git pull --ff-only || true
    [[ "$DRY_RUN" == "1" ]] || git push -u origin "$branch" || true
  )
}

ensure_fork_and_remote() {
  local src_org="$1" repo="$2" path="$3"
  ( cd "$path"
    if ! gh repo view "$DEST_ORG/$repo" >/dev/null 2>&1; then
      gh repo fork "$src_org/$repo" --org "$DEST_ORG" --clone=false || true
    fi
    git remote remove build 2>/dev/null || true
    local remote_url
    if [[ "$USE_HTTPS" == "1" ]]; then
      remote_url="https://github.com/$DEST_ORG/$repo.git"
    else
      remote_url="git@github.com:$DEST_ORG/$repo.git"
    fi
    git remote add build "$remote_url" || true
  )
}

push_branch_to_build() {
  local path="$1" branch="$2"
  ( cd "$path"
    if [[ "$DRY_RUN" == "0" ]]; then
      git push build "$branch:$branch" || true
    fi
  )
}

process_org() {
  local org="$1" branch="$2"
  for repo in $(gh repo list "$org" --limit 200 --json name -q '.[].name'); do
    log "Processing $org/$repo → $branch"
    local path
    path="$(clone_repo "$org" "$repo")"
    ensure_branch "$path" "$branch"
    ensure_fork_and_remote "$org" "$repo" "$path"
    push_branch_to_build "$path" "$branch"
  done
}

build_branch_merge() {
  local repo="$1"
  # choose an existing working tree for this repo
  local path=""
  [[ -d "$TMPDIR/$ORG_TEST/$repo/.git" ]] && path="$TMPDIR/$ORG_TEST/$repo"
  [[ -z "$path" && -d "$TMPDIR/$ORG_DEV/$repo/.git" ]] && path="$TMPDIR/$ORG_DEV/$repo"
  [[ -z "$path" ]] && return 0

  ( cd "$path"
    git fetch build --prune || true

    # start BUILD from TEST if available, else DEV
    if git ls-remote --exit-code --heads build TEST >/dev/null 2>&1; then
      git checkout -B BUILD build/TEST
    elif git ls-remote --exit-code --heads build DEV >/dev/null 2>&1; then
      git checkout -B BUILD build/DEV
    else
      log "[BUILD] Skip $repo (no TEST/DEV in fork)"
      return 0
    fi

    # merge the other branch if it exists
    if git ls-remote --exit-code --heads build DEV >/dev/null 2>&1; then
      git pull build DEV --allow-unrelated-histories --no-edit || true
    fi
    if git ls-remote --exit-code --heads build TEST >/dev/null 2>&1; then
      git pull build TEST --allow-unrelated-histories --no-edit || true
    fi

    [[ "$DRY_RUN" == "1" ]] || git push build BUILD:BUILD
    log "[BUILD] ${repo} updated"
  )
}

# ======== Main Function ========
main() {
  log "== SYNC TEST from $ORG_TEST =="
  process_org "$ORG_TEST" TEST

  log "== SYNC DEV from $ORG_DEV =="
  process_org "$ORG_DEV" DEV

  log "== BUILD =="
  # union of repo names across both orgs
  repos="$( (gh repo list "$ORG_TEST" --limit 200 --json name -q '.[].name'; \
             gh repo list "$ORG_DEV"  --limit 200 --json name -q '.[].name') | sort -u )"
  for r in $repos; do
    build_branch_merge "$r"
  done

  log "✅ Done."
}

# ======== Run ========
# Only run main function if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi