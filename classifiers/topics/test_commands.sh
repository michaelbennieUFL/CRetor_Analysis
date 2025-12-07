python compareModels.py \
  --model_name openai/inclusionAI/Ring-flash-2.0 \
  --api_base http://localhost:8080/v1 \
  --api_key dummy \
  --project_csv_path ./project-4-at-2025-12-05-03-06-66edb22a.csv \
  --topics_tsv_path ./predicted_topics_corrected.tsv \
  --optimized_prompt_path ./prompts/multi_shot.txt \
  --basic_prompt_path ./prompts/one_shot.txt
