#!/usr/bin/env bash
# Fetch flybody's fruitfly mesh assets (Apache-2.0, github.com/TuragaLab/flybody) that tools/assets.sh fills the flight model from.
# Pinned to one commit; unpacked into .work/flybody (gitignored). Overwrite the pin to upgrade.
# usage: tools/fetch_flybody.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=d015e9bfe441bd90ae431bac24c55cb74bdbce26
DEST=.work/flybody
ASSETS=$DEST/flybody/fruitfly/assets
ARCH=.work/fetch/flybody-$SHA.tar.gz

if [ -d "$ASSETS" ] && [ -n "$(ls -A "$ASSETS")" ]; then echo "ok $ASSETS (present)"; exit 0; fi
mkdir -p "$(dirname "$ARCH")" "$DEST"
curl -fL --retry 3 -o "$ARCH" "https://github.com/TuragaLab/flybody/archive/$SHA.tar.gz" \
  || { echo "FAIL fetch flybody @ $SHA: github.com/TuragaLab/flybody"; exit 1; }
tar -xzf "$ARCH" -C "$DEST" --strip-components=1 || { echo "FAIL unpack $ARCH"; exit 1; }
[ -n "$(ls -A "$ASSETS" 2>/dev/null)" ] || { echo "FAIL $ASSETS missing after unpack"; exit 1; }
echo "ok $ASSETS ($(ls "$ASSETS" | wc -l) files, Apache-2.0: $DEST/LICENSE)"
