python src/fine_tune.py train \
  --config="configs/train_with_think_tokens.yaml" \
  --data_file="data/code_switched_GAIR_LIMO_train_817.jsonl" \
  --output_dir="outputs/limo/code_switched-817-lora"

python src/fine_tune.py train \
  --config="configs/train_with_think_tokens.yaml" \
  --data_file="data/original_GAIR_LIMO_train_817.jsonl" \
  --output_dir="outputs/limo/original-817-lora"