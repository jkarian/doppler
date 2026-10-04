#!/usr/bin/env bash
# Dream several camera moves, then upscale each.
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/dream_and_upscale.sh scenes/canyon move-left move-right
set -e
HERE=$(dirname "$0")
SCENE=$1; shift
for MOVE in "$@"; do
  bash "$HERE/dream_views.sh" "$SCENE" "$MOVE" 2>&1 | grep -E '^== |Error' || true
  bash "$HERE/upscale_views.sh" "$SCENE" "$MOVE" 2>&1 | grep -E 'upscaled|Error' || true
done
