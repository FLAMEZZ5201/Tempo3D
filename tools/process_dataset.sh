#!/bin/bash
# Dataset preparation pipeline for TempoDetail.
# Converts raw 3D assets to the preprocessed format used for training:
#   1. Convert source meshes to GLB (convert_to_glb.py)
#   2. Render multi-view condition images via Blender (render.py)
#   3. Compute watertight mesh + SDF samples (watertight_and_sample.py)
#
# Usage:
#   bash tools/process_dataset.sh \
#       --src /path/to/raw_assets \
#       --glb /path/to/TempoDetail_glb \
#       --out /path/to/TempoDetail_preprocessed

set -e

# ---------- defaults ----------
SRC=""
GLB_DIR=""
OUT_DIR=""
RESOLUTION=512
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
BLENDER_BIN="blender"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --src)    SRC="$2";     shift 2 ;;
        --glb)    GLB_DIR="$2"; shift 2 ;;
        --out)    OUT_DIR="$2"; shift 2 ;;
        --res)    RESOLUTION="$2"; shift 2 ;;
        --blender) BLENDER_BIN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$SRC" || -z "$GLB_DIR" || -z "$OUT_DIR" ]]; then
    echo "Usage: bash tools/process_dataset.sh --src <raw> --glb <glb_dir> --out <out_dir>"
    exit 1
fi

mkdir -p "$GLB_DIR" "$OUT_DIR"

# ---------- step 1: convert to GLB ----------
echo "===== Step 1: Convert to GLB ====="
python3 "${TOOLS_DIR}/convert_to_glb.py" --src "$SRC" --dst "$GLB_DIR"

# ---------- step 2 & 3: render + watertight ----------
echo "===== Step 2 & 3: Render + Watertight ====="

TOTAL=0; SKIPPED=0; FAILED=0

for UID_DIR in "${GLB_DIR}"/*/; do
    [ -d "$UID_DIR" ] || continue
    UID=$(basename "$UID_DIR")
    TARGET="${OUT_DIR}/${UID}"

    if [ -d "${TARGET}/render_cond" ] && [ -d "${TARGET}/geo_data" ]; then
        SKIPPED=$((SKIPPED+1)); continue
    fi

    GLB_FILE=$(find "$UID_DIR" -maxdepth 1 -name "*.glb" | head -n 1)
    if [ -z "$GLB_FILE" ]; then
        echo "  [SKIP] No GLB found: $UID"; FAILED=$((FAILED+1)); continue
    fi

    mkdir -p "${TARGET}/render_cond" "${TARGET}/geo_data"

    echo "  [${UID}] Rendering..."
    "$BLENDER_BIN" --background --python "${TOOLS_DIR}/render.py" -- \
        --object "$GLB_FILE" \
        --output_folder "${TARGET}/render_cond" \
        --geo_mode \
        --resolution "$RESOLUTION" || { echo "  Render failed"; FAILED=$((FAILED+1)); continue; }

    PLY="${TARGET}/render_cond/mesh.ply"
    if [ ! -f "$PLY" ]; then
        echo "  [SKIP] mesh.ply not found: $UID"; FAILED=$((FAILED+1)); continue
    fi

    echo "  [${UID}] Watertight + SDF..."
    python3 "${TOOLS_DIR}/watertight_and_sample.py" \
        --input_obj "$PLY" \
        --output_prefix "${TARGET}/geo_data/${UID}" || { echo "  Watertight failed"; FAILED=$((FAILED+1)); continue; }

    echo "  [OK] $UID"
    TOTAL=$((TOTAL+1))
done

echo ""
echo "===== Done ====="
echo "Processed: $TOTAL  Skipped: $SKIPPED  Failed: $FAILED"