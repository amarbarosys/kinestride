#!/usr/bin/env bash
# CPU smoke, no GPU, no .work assets: doc links, then export the in-git spider test body and build the full step on the CPU.
# Catches export and Mojo compile errors; GPU-only link errors (e.g. atan2f) still need a check.sh run.
# usage: tools/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tools/doclinks.sh
BUILD_ONLY=1 .venv/bin/python -m kinestride.model check kinestride/models/spider_test.xml .work/smoke step 64 > .work/smoke.log 2>&1 \
  || { echo "FAIL smoke build: .work/smoke.log"; tail -5 .work/smoke.log; exit 1; }
echo "PASS smoke ($(tail -1 .work/smoke.log))"
