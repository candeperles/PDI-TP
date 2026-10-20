#!/usr/bin/env bash
# Downloads the grupo_5 dataset from the GitHub Release and extracts it into data/.
# Usage: ./scripts/download_data.sh
set -euo pipefail

REPO="candeperles/PDI-TP"
TAG="dataset-v1"
ASSET="grupo_5.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data"
ZIP_PATH="$DATA_DIR/$ASSET"

mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/grupo_5" ]; then
  echo "data/grupo_5 already exists, skipping download. Delete it first to re-download."
  exit 0
fi

echo "Downloading $ASSET from $REPO release $TAG..."
if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir "$DATA_DIR" --clobber
else
  curl -L -o "$ZIP_PATH" "https://github.com/$REPO/releases/download/$TAG/$ASSET"
fi

echo "Extracting..."
unzip -q "$ZIP_PATH" -d "$DATA_DIR"
rm "$ZIP_PATH"

echo "Done. Dataset is in $DATA_DIR/grupo_5"
