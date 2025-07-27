srun --partition a01 --gres=gpu:1 --pty "bash"

jupyter notebook  --notebook-dir=/home/fit/renju/WORK/lxm  --ip=0.0.0.0 --port=10059

srun --partition=a01 --gres=gpu:1  --job-name=predict --kill-on-bad-exit=1 --output=/home/fit/renju/WORK/lxm/log.out python /home/fit/renju/WORK/lxm/Compression/quantization.py
conda activate lxm_infer

huggingface-cli download --repo-type dataset --resume-download tiiuae/falcon-refinedweb  --local-dir /home/fit/renju/WORK/lxm/datasets/falcon-refinedweb

huggingface-cli download --resume-download ByteDance-Seed/UI-TARS-2B-SFT --local-dir /home/fit/renju/WORK/lxm/models/UI-TARS-2B-SFT


cinfo -p AI4Good_S1 occupy-reserved
scontrol show job  17607627 -d
srun -p a01 --nodes=1 --gres=gpu:1 jupyter notebook  --notebook-dir=/home/fit/renju/WORK/lxm  --ip=0.0.0.0 --port=10059

python convert_hf_to_gguf.py /home/fit/renju/WORK/lxm/models/test --outfile /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave.gguf
python -m pip install --upgrade -r requirements/requirements-convert_hf_to_gguf.txt

/home/fit/renju/WORK/lxm/llama.cpp/build/bin/llama-cli -m /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave-Q4_K_M.gguf -no-cnv --prompt "Once upon a time"
gdb --args /home/fit/renju/WORK/lxm/llama.cpp/build/bin/llama-cli -m /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave-Q4_K_M.gguf -no-cnv --prompt "Once upon a time"

/home/fit/renju/WORK/lxm/llama.cpp/build/bin/llama-quantize /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave.gguf /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave-Q4_K_M.gguf Q4_K_M
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF
cmake --build build -j 100
gdb --args /home/fit/renju/WORK/lxm/llama.cpp/build/bin/llama-cli -m /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave-Q4_K_M.gguf -no-cnv --prompt "Once upon a time"
tar -xzf /home/fit/renju/WORK/lxm/eval.tar.gz -C /home/fit/renju/WORK/miniconda3/envs/lxm_infer

scp -P 9207  /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite_Wave-Q4_K_M.gguf wangtuowei@yutong.time-crystal.org:/home/wangtuowei/lxm/models/


accelerate launch Predict/finetune.py \
    --model_name_or_path /home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite \
    --output_dir ./output \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5

srun --partition=a01 --gres=gpu:2  --job-name=test_cats --kill-on-bad-exit=1 --output=/home/fit/renju/WORK/lxm/CATS/log.out bash /home/fit/renju/WORK/lxm/CATS/reproduction_script.sh /home/fit/renju/WORK/lxm/CATS/t_ckpt/ /home/fit/renju/WORK/lxm/CATS/t_res/
