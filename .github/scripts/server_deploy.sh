#!/usr/bin/env bash
# Runs on the cPanel server over SSH (`ssh server 'bash -s -- <command>' < this file`), called by
# .github/workflows/deploy.yml:
#   deploy           fast-forward the checkout to origin/main and restart the app
#   startup-ok       check the app's latest start loaded every service (/health answers even when they didn't)
#   rollback <sha>   put the checkout back to <sha> and restart the app
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/vision}"
PYTHON="$HOME/virtualenv/vision/3.13/bin/python"
cd "$APP_DIR"

restart() {
  # Stop this app's lswsgi processes; LiteSpeed starts a fresh one, running the new code, on the next request.
  # Bash built-ins only, so this still works when the account is at its process limit and `ps`/`pkill` can't start.
  local p c x
  for p in /proc/[0-9]*; do
    c=""
    { while IFS= read -r -d '' x; do c="$c $x"; done < "$p/cmdline"; } 2>/dev/null || continue
    if [[ $c == *lswsgi* && $c == *"$APP_DIR/"* ]]; then
      kill "${p#/proc/}" 2>/dev/null || true
    fi
  done
}

case "${1:-}" in
  deploy)
    if ! git diff --quiet || ! git diff --cached --quiet; then
      echo "Not deploying: $APP_DIR has uncommitted changes to tracked files:"
      git status --short
      exit 1
    fi
    before=$(git rev-parse HEAD)
    git fetch --quiet origin main
    # Fast-forward only: a merge that conflicts would leave conflict markers in the live code
    if ! git merge --ff-only --quiet origin/main; then
      echo "Not deploying: the server checkout has commits that are not on GitHub."
      echo "Make it match origin/main once (vision_service_production_fix.md, section 7), then re-run."
      exit 1
    fi
    after=$(git rev-parse HEAD)
    if ! git diff --quiet "$before" "$after" -- requirements.txt; then
      echo "requirements.txt changed; installing packages"
      # Thread caps: an uncapped Python on this 96-core server can push the account past its process limit
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" -m pip install --quiet -r requirements.txt
    fi
    restart
    echo "Deployed ${before:0:7} -> ${after:0:7}"
    ;;

  startup-ok)
    last=$(awk '/--- STARTUP at/ { block = "" } { block = block $0 "\n" } END { printf "%s", block }' wsgi_error.log)
    if ! grep -q "All services loaded" <<<"$last" || grep -q "IMPORT ERROR" <<<"$last"; then
      echo "The app's latest start did not load its services:"
      tail -n 40 <<<"$last"
      exit 1
    fi
    sed -n '1,3p' <<<"$last"
    ;;

  rollback)
    target="${2:?usage: rollback <sha>}"
    if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$target")" ]; then
      echo "Already at ${target:0:7}; nothing to roll back"
      exit 0
    fi
    git reset --hard --quiet "$target"
    restart
    echo "Rolled back to ${target:0:7}"
    ;;

  *)
    echo "usage: server_deploy.sh deploy | startup-ok | rollback <sha>" >&2
    exit 2
    ;;
esac
