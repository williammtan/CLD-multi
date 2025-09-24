 python3 "cronos_trainer.py" \
      --model_name whisper-small \
      --data_dir "data/final_dry" \
      --output_dir "models/cvx_dry"

python3 "benchmark_cld.py" \
        --dataset_path "data/final_dry" \
        --whisper_path "openai/whisper-small" \
        --cld_path "models/cvx_dry/whisper-small_trained_cvx_mlp.pkl" \
        --cld_type cvx \
        --batch_size 8