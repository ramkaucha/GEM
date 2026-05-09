cd "$(dirname "$0")"
PYTHONPATH="$(pwd):${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python llava/eval/model_ecg_resume.py \
  --model-path "/home/ram/work/storage/GEM" \
  --image-folder "/home/ram/work/storage/sample_ecg" \
  --question-file "/home/ram/work/GEM/questions.json" \
  --answers-file "/home/ram/work/GEM/answer_img2.jsonl" \
  --conv-mode llava_v1 \
  --ecg-folder "/home/ram/work/storage/sample_ecg" \
  --temperature 0 \
  --open_clip_config coca_ViT-B-32 \
  --num_beams 1 \
  --max_new_tokens 512 \
  --modality image
