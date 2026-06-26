#!/usr/bin/env bash
# cleanup_sam3.sh - remove SAM3 model weights, repo, and related caches
set -e

echo "This will delete:"
echo "  - SAM3 checkpoint cache (~3.5GB): ~/.cache/huggingface/hub/models--facebook--sam3"
echo "  - SAM3 cloned repo:               ./sam3_repo"
echo ""
read -p "Continue? [y/N] " confirm
[[ "$confirm" == [yY] ]] || { echo "Aborted."; exit 0; }

# 1. Model weights from HF cache
rm -rf ~/.cache/huggingface/hub/models--facebook--sam3
rm -rf ~/.cache/huggingface/hub/.locks/models--facebook--sam3
rm -rf ./data
# 2. Uninstall the package, then delete the cloned repo
pip uninstall -y sam3 2>/dev/null || true
rm -rf "$(dirname "$0")/sam3_repo"

echo ""
echo "Done. Space reclaimed:"
df -h ~ | tail -1