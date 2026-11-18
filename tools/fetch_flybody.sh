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

if [ -d "$ASSETS" ] && [ -n "$(ls -A "$ASSETS")" ]; then :; else
mkdir -p "$(dirname "$ARCH")" "$DEST"
curl -fL --retry 3 -o "$ARCH" "https://github.com/TuragaLab/flybody/archive/$SHA.tar.gz" \
  || { echo "FAIL fetch flybody @ $SHA: github.com/TuragaLab/flybody"; exit 1; }
tar -xzf "$ARCH" -C "$DEST" --strip-components=1 || { echo "FAIL unpack $ARCH"; exit 1; }
[ -n "$(ls -A "$ASSETS" 2>/dev/null)" ] || { echo "FAIL $ASSETS missing after unpack"; exit 1; }
fi
echo "ok $ASSETS ($(ls "$ASSETS" | wc -l) files, Apache-2.0: $DEST/LICENSE)"

# Flight wing pattern (wing_pattern_fmech.npy, flybody flight-imitation dataset, figshare 10.25378/janelia.25309105).
WPG=.work/flybody-data/wing_pattern_fmech.npy; WSHA=f97b975ef1b5adbe42c208ea9665f3c37fa432914f56f3943f1697cdb090d1ce
if [ "$(sha256sum "$WPG" 2>/dev/null | cut -c1-64)" != "$WSHA" ]; then
  Z=.work/fetch/flight-imitation.zip; mkdir -p "$(dirname "$WPG")"
  curl -fL --retry 3 -o "$Z" https://ndownloader.figshare.com/files/51196859 || { echo "FAIL fetch flight-imitation dataset (figshare 51196859)"; exit 1; }
  unzip -o -q -j "$Z" wing_pattern_fmech.npy -d "$(dirname "$WPG")" || { echo "FAIL unzip wing_pattern_fmech.npy from $Z"; exit 1; }
  [ "$(sha256sum "$WPG" | cut -c1-64)" = "$WSHA" ] || { echo "FAIL $WPG sha256 mismatch"; exit 1; }
fi
echo "ok $WPG"
