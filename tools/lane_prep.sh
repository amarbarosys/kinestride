#!/usr/bin/env bash
# lane_prep.sh <worktree> <item> <repo>: lane-dispatch LANE_PREP for kinestride.
# Links the venv and the shared read-only caches (test bodies, MuJoCo source, port receipts) from the main checkout.
set -euo pipefail
wt=$1; repo=$3
[ -e "$wt/.venv" ] || ln -s "$repo/.venv" "$wt/.venv"
mkdir -p "$wt/.work"
for d in assets assets-mojo mujoco-src spider-body; do
  [ -e "$wt/.work/$d" ] || ln -s "$repo/.work/$d" "$wt/.work/$d"
done
mkdir -p "$wt/.work/port" "$wt/.work/briefs"
cp -n "$repo"/.work/port/*.c "$repo"/.work/port/*.fields.md "$wt/.work/port/" 2>/dev/null || true
