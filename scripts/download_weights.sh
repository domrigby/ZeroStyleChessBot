#!/usr/bin/env bash
# Download the pretrained network into neural_nets/ (run from anywhere).
set -euo pipefail

# TODO(release): placeholder until the GitHub Release exists. Create release WEIGHTS_TAG on
# github.com/domrigby/ZeroStyleChessBot with example_network.pt attached as an asset.
WEIGHTS_TAG="weights-v1"
ASSET_NAME="example_network.pt"
URL="https://github.com/domrigby/ZeroStyleChessBot/releases/download/${WEIGHTS_TAG}/${ASSET_NAME}"

DESTINATION="$(cd "$(dirname "$0")/.." && pwd)/neural_nets/${ASSET_NAME}"

if [ -f "${DESTINATION}" ]; then
    echo "${DESTINATION} already exists, skipping download."
    exit 0
fi

echo "Downloading ${URL}"
curl --fail --location --output "${DESTINATION}.part" "${URL}"
mv "${DESTINATION}.part" "${DESTINATION}"
echo "Saved to ${DESTINATION}"
