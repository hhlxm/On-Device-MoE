#!/bin/bash
###
 # @Author: hhlxm 578723542@qq.com
 # @Date: 2025-05-23 17:10:12
 # @LastEditors: hhlxm 578723542@qq.com
 # @LastEditTime: 2025-05-24 15:34:22
 # @FilePath: /lxm/Analyse/scripts/run_mixtral.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

MODEL_NAME="/home/fit/renju/WORK/lxm/models/Mixtral_8x7B_v0_1"
MODEL_TYPE="Mixtral"
# CACHE_RATIOS=(0.125 0.25 0.5 0.75)
# DATASETS=("alpaca-zh" "humaneval" "gsm8k" "swag" "squad")
CACHE_RATIOS=(0.125 0.25 0.5 0.75)
PRE_RATIOS=(0.5 2.0 3.0)
DATASETS=("alpaca")
# OUTPUT_DIR="/home/fit/renju/WORK/lxm/Analyse/results_hot/figures"
PYTHON_SCRIPT="/home/fit/renju/WORK/lxm/Analyse/eval.py"

generate_config() {
    local dataset=$1
    local cache_ratio=$2
    local pre_ratio=$3
    cat << EOF
model:
  name: "$MODEL_NAME"
  type: "$MODEL_TYPE"
  device: "cuda"
  cache_ratio: $cache_ratio
  pre_ratio: $pre_ratio

dataset:
  name: "$dataset"
  sample_size: 100

generation:
  max_new_tokens: 200

output:
  dir: "/home/fit/renju/WORK/lxm/Analyse/results_ab_pre_ratio/pre_ratio_$pre_ratio/figures"

analysis:
  layer_start: 0
  layer_end: 32
  token_start: 1
  token_end: 10000

matrix:
  num_layers: 32
  num_experts: 8
EOF
}

# ==================== 主执行逻辑 ====================
echo "开始mixtral模型实验..."

for dataset in "${DATASETS[@]}"; do
    for cache_ratio in "${CACHE_RATIOS[@]}"; do
      for pre_ratio in "${PRE_RATIOS[@]}"; do
          echo "运行: $dataset, Cache_Ratio=$cache_ratio pre_ratio=$pre_ratio"
          
          # 生成临时配置文件并运行
          config_file="./mixtral_${dataset}_${cache_ratio}_${pre_ratio}.yaml"
          generate_config "$dataset" "$cache_ratio" "$pre_ratio"> "$config_file"
          sbatch_script="./sbatch_mixtral_${dataset}_${cache_ratio}_${pre_ratio}.sh"
          cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH -J mixtral_${dataset}_${cache_ratio}_${pre_ratio}
#SBATCH -o ./log/mixtral_${dataset}_${cache_ratio}_%j.out
#SBATCH -e ./log/%j_stderr_${dataset}
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
source /home/fit/renju/WORK/miniconda3/envs/lxm_eval/bin/activate
conda activate lxm_eval
python $PYTHON_SCRIPT $config_file
rm -f $config_file
EOF
          sbatch "$sbatch_script"
          rm -f "$sbatch_script"
        done
    done
done

echo "实验完成！"