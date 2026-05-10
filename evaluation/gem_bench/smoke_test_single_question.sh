#!/bin/bash
# Smoke test for single-question inference. Designed to be run inside an
# interactive PBS session with 1 GPU and ~15 min walltime, e.g.:
#   qsub -I -l select=1:ncpus=4:mem=16gb:ngpus=1 -l walltime=00:15:00
#
# What it does:
#   1. Symlinks the first N records (default 50) into a small temp folder.
#   2. Runs all 3 modalities sequentially on that subset on GPU 0.
#   3. Runs the evaluation step against metadata_with_description.csv.
#
# Failures (wrong model path, missing ECG tower, OOM, missing .hea/.dat) will
# surface within ~5 minutes instead of 6 hours.
#
# Edit the four paths below for your environment, then:
#   bash smoke_test_single_question.sh

set -euo pipefail

# ---- EDIT THESE FOUR ---------------------------------------------------------
GEM_ROOT=~/ra/GEM
CONDA_ACTIVATE=/srv/scratch/z5367751/miniconda3/bin/activate
CONDA_ENV=gem
MODEL_PATH=/srv/scratch/z5367751/checkpoints/GEM-7B
ECG_TOWER=/srv/scratch/z5367751/checkpoints/ecg_coca/cpt_wfep_epoch_20.pt
RECORDS_FULL=/srv/scratch/z5367751/records_img
# ------------------------------------------------------------------------------

N_SAMPLES=${N_SAMPLES:-50}
SMOKE_DIR=${SMOKE_DIR:-/tmp/gem_smoke_${USER}_$$}
RECORDS_SMOKE="${SMOKE_DIR}/records"
OUT_DIR="${SMOKE_DIR}/out"

echo "[smoke] GEM_ROOT=$GEM_ROOT"
echo "[smoke] MODEL_PATH=$MODEL_PATH"
echo "[smoke] ECG_TOWER=$ECG_TOWER"
echo "[smoke] RECORDS_FULL=$RECORDS_FULL"
echo "[smoke] N_SAMPLES=$N_SAMPLES"
echo "[smoke] SMOKE_DIR=$SMOKE_DIR"

# Sanity-check inputs early — better to fail here than after model load.
[[ -d "$GEM_ROOT" ]]      || { echo "GEM_ROOT not found: $GEM_ROOT" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]]    || { echo "MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }
[[ -f "$ECG_TOWER" ]]     || echo "[warn] ECG_TOWER not found at $ECG_TOWER (ok if config.json already has it)"
[[ -d "$RECORDS_FULL" ]]  || { echo "RECORDS_FULL not found: $RECORDS_FULL" >&2; exit 1; }

# Activate env
# shellcheck disable=SC1090
source "$CONDA_ACTIVATE" "$CONDA_ENV"

cd "$GEM_ROOT"

# Build a small smoke records folder by symlinking the first N record ids that
# have *all three* required files (.hea, .dat, .png). Symlinks are free.
mkdir -p "$RECORDS_SMOKE" "$OUT_DIR"
echo "[smoke] Selecting $N_SAMPLES records that have .hea + .dat + .png..."
count=0
for hea in "$RECORDS_FULL"/*.hea; do
  rid=$(basename "$hea" .hea)
  if [[ -f "$RECORDS_FULL/$rid.dat" && -f "$RECORDS_FULL/$rid.png" ]]; then
    ln -sf "$RECORDS_FULL/$rid.hea" "$RECORDS_SMOKE/$rid.hea"
    ln -sf "$RECORDS_FULL/$rid.dat" "$RECORDS_SMOKE/$rid.dat"
    ln -sf "$RECORDS_FULL/$rid.png" "$RECORDS_SMOKE/$rid.png"
    count=$((count+1))
    if (( count >= N_SAMPLES )); then break; fi
  fi
done
echo "[smoke] Linked $count records into $RECORDS_SMOKE"
if (( count == 0 )); then
  echo "[smoke] No records had all three of .hea/.dat/.png — check $RECORDS_FULL" >&2
  exit 1
fi

# Run each modality on GPU 0, sequentially. Output one JSONL per modality.
for MOD in image ecg both; do
  echo "----- [smoke] modality=$MOD -----"
  CUDA_VISIBLE_DEVICES=0 python llava/eval/model_ecg_resume.py \
    --model-path "$MODEL_PATH" \
    --single-question \
    --records-folder "$RECORDS_SMOKE" \
    --modality "$MOD" \
    --answers-file "$OUT_DIR/answers_${MOD}.jsonl" \
    --conv-mode llava_v1 \
    --ecg_tower "$ECG_TOWER" \
    --open_clip_config coca_ViT-B-32 \
    2>&1 | tee "$OUT_DIR/${MOD}.log"
done

# Combine and evaluate.
cat "$OUT_DIR"/answers_*.jsonl > "$OUT_DIR/answers.jsonl"
echo "[smoke] Combined JSONL -> $OUT_DIR/answers.jsonl"
echo "[smoke] Total rows: $(wc -l < "$OUT_DIR/answers.jsonl")"

python evaluation/eval_single_question.py \
  --answers-file "$OUT_DIR/answers.jsonl" \
  --metadata-csv preprocess/reference/metadata_with_description.csv \
  --output-csv  "$OUT_DIR/eval.csv" \
  --threshold 70

echo
echo "[smoke] DONE. Artifacts in: $OUT_DIR"
echo "[smoke] Inspect a sample row with:"
echo "        head -1 $OUT_DIR/answers.jsonl | python -m json.tool"
