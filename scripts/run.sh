CUDA_VISIBLE_DEVICES=2,3,4,5 \
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=/home/pairshoe/lxm/models/Qwen1.5-MoE-A2.7B,dtype=float16,trust_remote_code=True,parallelize=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --num_fewshot 5 \
    --output_path ./results/Qwen1.5-MoE-A2.7B-gsm8k \
    --log_samples