#!/usr/bin/env bash
# Blender -> MJCF -> engine: builds a two-part rig in headless Blender, exports it with the blender-mjcf add-on
# (Balint-H/blender-mjcf, MIT, pinned), then model.py check step vs MuJoCo. Needs blender on PATH and the GPU.
# usage: gpu-wait run --timeout 1200 -- tools/blender_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=3fb51ccfe559f66714601180f8268f8587e442af
W=.work/blender; P=$W/mjcf_export_plugin-$SHA.py; mkdir -p $W
[ -s $P ] || gh api "repos/Balint-H/blender-mjcf/contents/mjcf_export_plugin.py?ref=$SHA" -q .content | base64 -d > $P \
  || { echo "FAIL fetch plugin: gh api"; rm -f $P; exit 1; }
blender -b --factory-startup -P tools/blender_rig.py -- $P $W/rig > $W/blender.log 2>&1 && [ -s $W/rig/rig_floor.xml ] \
  || { echo "FAIL blender export: $W/blender.log"; tail -5 $W/blender.log; exit 1; }
.venv/bin/python -m kinestride.model check $W/rig/rig_floor.xml $W/chk step > $W/check.log 2>&1 && grep -q '^PASS step' $W/check.log \
  || { echo "FAIL rig step check: $W/check.log"; grep -E '^FAIL' $W/check.log | head -5; exit 1; }
X=$(.venv/bin/python -c "from kinestride import blend; print(blend.resolve('$W/rig/rig.blend'))") \
  || { echo "FAIL blend resolve: $W/rig/rig.blend"; exit 1; }
cmp -s "$X" $W/rig/rig.xml || { echo "FAIL blend resolve: $X differs from $W/rig/rig.xml"; diff "$X" $W/rig/rig.xml | head -5; exit 1; }
.venv/bin/python -m kinestride.model check $W/rig/rig.blend $W/chkb step > $W/checkb.log 2>&1 && grep -q '^PASS step' $W/checkb.log \
  || { echo "FAIL .blend step check: $W/checkb.log"; grep -E '^FAIL' $W/checkb.log | head -5; exit 1; }
BL=/nonexistent .venv/bin/python -c "from kinestride import blend; blend.resolve('$W/rig/rig.blend')" \
  || { echo "FAIL blend cache: second load needed Blender"; exit 1; }
echo "PASS blender rig ($(grep '^PASS step' $W/check.log)), .blend direct ($(grep '^PASS step' $W/checkb.log)), cache hit"
