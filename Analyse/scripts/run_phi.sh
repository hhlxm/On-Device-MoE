#!/bin/bash

MODEL_NAME="/home/fit/renju/WORK/lxm/models/Phi_3_5_MoE_instruct"
MODEL_TYPE="PhiMoE"
CACHE_RATIOS=(0.125 0.25 0.5 0.75)
DATASETS=("alpaca" "alpaca-zh" "humaneval" "gsm8k" "swag" "squad")
# CACHE_RATIOS=(0.5)
# DATASETS=("alpaca")
OUTPUT_DIR="/home/fit/renju/WORK/lxm/Analyse/results_hot/figures"
PYTHON_SCRIPT="/home/fit/renju/WORK/lxm/Analyse/eval.py"

generate_config() {
    local dataset=$1
    local cache_ratio=$2
    cat << EOF
model:
  name: "$MODEL_NAME"
  type: "$MODEL_TYPE"
  device: "cuda"
  cache_ratio: $cache_ratio

dataset:
  name: "$dataset"
  sample_size: 100

generation:
  max_new_tokens: 200

output:
  dir: "$OUTPUT_DIR"

analysis:
  layer_start: 0
  layer_end: 32
  token_start: 1
  token_end: 10000

matrix:
  num_layers: 32
  num_experts: 16
EOF
}

# ==================== 主执行逻辑 ====================
echo "开始Phi模型实验..."

for dataset in "${DATASETS[@]}"; do
    for cache_ratio in "${CACHE_RATIOS[@]}"; do
        echo "运行: $dataset, Cache_Ratio=$cache_ratio"
        
        # 生成临时配置文件并运行
        config_file="./phi_${dataset}_${cache_ratio}.yaml"
        generate_config "$dataset" "$cache_ratio" > "$config_file"
        sbatch_script="./sbatch_phi_${dataset}_${cache_ratio}.sh"
        cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH -J phi_${dataset}_${cache_ratio}
#SBATCH -o ./log/phi_${dataset}_${cache_ratio}_%j.out
#SBATCH -e ./log/%j_stderr_${dataset}
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
source /home/fit/renju/WORK/miniconda3/envs/lxm_infer/bin/activate
conda activate lxm_infer
python $PYTHON_SCRIPT $config_file
rm -f $config_file
EOF
        sbatch "$sbatch_script"
        rm -f "$sbatch_script"
    done
done

echo "实验完成！"