#!/bin/bash
# Populate the transformers_modules cache with jina custom code for offline use.
# transformers caches trust_remote_code .py files here and sometimes copies them
# incompletely. Run this after the modules cache is cleared, or after a
# transformers upgrade -- the folder-naming scheme can change (e.g. transformers
# 4.57 encodes "-" as "_hyphen_": jina-embeddings-v3 -> jina_hyphen_embeddings_hyphen_v3).
# This script matches cache folders by glob, so it works across naming schemes.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==== EDIT THESE PATHS to where the jina folders actually live ====
JINA_CLIP_V2_DIR="${JINA_CLIP_V2_DIR:-$SCRIPT_DIR/../src/jina-clip-v2}"
JINA_EMB_V3_DIR="${JINA_EMB_V3_DIR:-$SCRIPT_DIR/../src/jina-embeddings-v3}"

TM_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/modules/transformers_modules"

# fill <source dir> <cache-folder glob>
fill() {
    local src_dir="$1"
    local glob="$2"
    if [ ! -d "$src_dir" ]; then
        echo "[SKIP] source not found: $src_dir"
        return
    fi
    local hit=0
    for dst in "$TM_CACHE"/$glob; do
        [ -d "$dst" ] || continue
        cp "$src_dir"/*.py "$dst"/
        echo "[OK] $(basename "$src_dir") -> $dst ($(ls "$dst"/*.py | wc -l) .py files)"
        hit=1
    done
    if [ "$hit" = 0 ]; then
        # No cache folder yet -- create the transformers-4.57-style name as a best guess.
        local guess="$TM_CACHE/$(basename "$src_dir" | sed 's/-/_hyphen_/g')"
        mkdir -p "$guess"
        cp "$src_dir"/*.py "$guess"/
        echo "[NEW] no existing folder matched -- created $guess"
        echo "      (if training still fails, run it once so transformers creates"
        echo "       the real folder, then re-run this script)"
    fi
}

mkdir -p "$TM_CACHE"
fill "$JINA_CLIP_V2_DIR" '*jina*clip*v2*'
fill "$JINA_EMB_V3_DIR" '*jina*embeddings*v3*'
echo "[DONE] offline HF custom modules populated"
