#!/usr/bin/env bash
# Test bodies for the check.sh gates, kept out of git (11 MB fly, 135 MB flight model, MuJoCo sources for reading).
# Copies them into .work/ from a source checkout that already built them (default: fly-connectome).
# usage: tools/assets.sh [SRC]   -> .work/assets/fly/body.xml, .work/assets/flight/flight.xml, .work/mujoco-src/
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:-$HOME/Projects/science/fly-connectome}; fail=0
copy() { [ -e "$1" ] || { echo "FAIL asset $2: $1 missing"; fail=$((fail+1)); return; }; mkdir -p "$(dirname "$2")"; rm -rf "$2"; cp -r "$1" "$2"; echo "ok $2 ($(du -sh "$2" | cut -f1))"; }
copy "$S/.work/agent/gpu-body-port/body-v2" .work/assets/fly
copy "$S/.work/flight-body" .work/assets/flight
copy "$S/.work/agent/body_engine/mujoco-src" .work/mujoco-src
.venv/bin/python -m kinestride.models.spider .work/spider-body > .work/spider-body.log 2>&1 && echo "ok .work/spider-body (generated)" || { echo "FAIL spider body: .work/spider-body.log"; fail=$((fail+1)); }
[ $fail -eq 0 ] || { echo "FAIL $fail assets"; exit 1; }
