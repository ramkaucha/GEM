CUDA_VISIBLE_DEVICES=0 python interactive_gem_cli.py \
  --model-path "/media/tower1/DATA21/Ram/models/GEM" \
  --image-path "/media/tower1/DATA21/Ram/sample/p1001/p10018081/s42149331/42149331.png" \
  --ecg-record "/media/tower1/DATA21/Ram/sample/p1001/p10018081/s42149331/42149331" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 512
