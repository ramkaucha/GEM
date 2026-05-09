#!/bin/bash
# Run single-question inference over a records folder for all three modalities,
# appending into one combined JSONL. Then optionally run evaluation.
#
# Required:
#   -m  model_name      (subfolder under CKPT_DIR, e.g. GEM-7B)
#   -r  records_folder  (folder containing <id>.hea/.dat/.png triples)
#
# Optional:
#   -c  ckpt_dir        (default: empty — set CKPT_DIR env or pass -c)
#   -e  ecg_tower_path  (default: empty)
#   -o  open_clip_cfg   (default: coca_ViT-B-32)
#   -s  save_dir        (default: ../../eval_outputs/<model>/single_question)
#   -t  threshold       (default: 70)
#   -k  skip_eval       (set anything; skips the evaluation step)
#
# Usage:
#   bash bench_single_question.sh -m GEM-7B -r /data/records

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-}"
ECG_TOWER=""
OPEN_CLIP_CONFIG="coca_ViT-B-32"
SAVE_DIR_OVERRIDE=""
THRESHOLD=70
SKIP_EVAL=""

while getopts "m:r:c:e:o:s:t:k:h" option; do
  case "$option" in
    m) MODEL_NAME="$OPTARG" ;;
    r) RECORDS_FOLDER="$OPTARG" ;;
    c) CKPT_DIR="$OPTARG" ;;
    e) ECG_TOWER="$OPTARG" ;;
    o) OPEN_CLIP_CONFIG="$OPTARG" ;;
    s) SAVE_DIR_OVERRIDE="$OPTARG" ;;
    t) THRESHOLD="$OPTARG" ;;
    k) SKIP_EVAL=1 ;;
    h)
      echo "Usage: $0 -m model_name -r records_folder [-c ckpt_dir] [-e ecg_tower] [-o open_clip_cfg] [-s save_dir] [-t threshold] [-k]"
      exit 0
      ;;
    *) echo "Unknown option"; exit 1 ;;
  esac
done

if [[ -z "${MODEL_NAME:-}" || -z "${RECORDS_FOLDER:-}" ]]; then
  echo "Error: -m model_name and -r records_folder are required. Use -h for help." >&2
  exit 1
fi

MODEL_PATH="${CKPT_DIR}/${MODEL_NAME}"
SAVE_DIR="${SAVE_DIR_OVERRIDE:-../../eval_outputs/${MODEL_NAME}/single_question}"
ANSWERS_FILE="${SAVE_DIR}/answers.jsonl"
EVAL_CSV="${SAVE_DIR}/eval.csv"

mkdir -p "$SAVE_DIR"

echo "[bench_single_question] model_path=$MODEL_PATH"
echo "[bench_single_question] records_folder=$RECORDS_FOLDER"
echo "[bench_single_question] answers_file=$ANSWERS_FILE"

# Run inference once per modality. The python script dedupes by (question_id, modality)
# so re-running is safe and resumable.
for MODALITY in image ecg both; do
  echo "----- Modality: ${MODALITY} -----"
  CUDA_VISIBLE_DEVICES=0 python ../../llava/eval/model_ecg_resume.py \
    --model-path "$MODEL_PATH" \
    --single-question \
    --records-folder "$RECORDS_FOLDER" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode "llava_v1" \
    --modality "$MODALITY" \
    --ecg_tower "$ECG_TOWER" \
    --open_clip_config "$OPEN_CLIP_CONFIG"
done

if [[ -n "$SKIP_EVAL" ]]; then
  echo "[bench_single_question] Skipping evaluation (-k set)."
  exit 0
fi

echo "----- Evaluation -----"
python ../eval_single_question.py \
  --answers-file "$ANSWERS_FILE" \
  --metadata-csv "../../preprocess/reference/metadata_with_description.csv" \
  --output-csv "$EVAL_CSV" \
  --threshold "$THRESHOLD"
