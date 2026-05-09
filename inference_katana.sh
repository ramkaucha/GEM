cd "$(dirname "$0")"
PYTHONPATH="$(pwd):${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python llava/eval/model_ecg_resume.py \
  --model-path "/srv/scratch/z5367751/GEM" \
  --image-folder "/srv/scratch/z5367751/sample_ecg" \
  --question-file "/home/z5367751/ra/GEM/questions.json" \
  --answers-file "/home/z5367751/ra/GEM/answer_img2.jsonl" \
  --conv-mode llava_v1 \
  --ecg-folder "/srv/scratch/z5367751/sample_ecg" \
  --temperature 0 \
  --open_clip_config coca_ViT-B-32 \
  --num_beams 1 \
  --max_new_tokens 512 \
  --modality image
