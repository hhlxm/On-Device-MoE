MODEL_NAME="/home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite"
# CACHE_RATIOS=(0.125 0.25 0.5 0.75)
# DATASETS=("alpaca" "alpaca-zh" "humaneval" "gsm8k" "swag" "squad")
DATASETS=("alpaca")
PRE_DIS=1
OUTPUT_DIR="/home/fit/renju/WORK/lxm/Predict/results/DeepSeek_V2_Lite/output_pregate_instruct_finetune_linear_dis$PRE_DIS"
PYTHON_SCRIPT="/home/fit/renju/WORK/lxm/Predict/train.py"
TYPE="pregate" # pregate/pretoken
CHECKPOINT_PATH="/home/fit/renju/WORK/lxm/Predict/results/DeepSeek_V2_Lite/output_pregate_instruct_finetune_linear/deepseek_v2_lite/alpaca/alpaca/checkpoints/checkpoint_epoch_1.pt"
generate_config() {
    local dataset=$1
    local output_dir=$2
    local type=$3
    local checkpoint_path=$4
    local pre_dis=$5
    cat << EOF
# 随机种子
seed: 2025

# 模型配置
model:
  name: "/home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite"
  type: ${type}
  pre_dis: ${pre_dis}  # 预训练距离
  
# 输出配置
output:
  dir: ${output_dir}  # 输出目录

# 训练配置
train:
  num_epochs: 1          # 训练轮数
  print_every: 10        # 每多少步打印一次训练损失
  eval_every: 10          # 每多少步进行一次验证
  train_max_steps: -1    # -1表示自适应
  val_max_steps: 2       # 验证最大步数
  checkpoint_path: ${checkpoint_path}  # 检查点路径
  optimizer:
    lr: 0.1             # 学习率
    eps: 0.0001            # Adam 的 epsilon 参数

# 数据配置
data:
  train_dataset_name: "alpaca"  # 训练数据集名称
  test_dataset_name: ${dataset}               # 测试数据集名称
  train_type: "train"                     # 训练数据集类型
  test_type: "train"                 # 测试数据集类型
  batch_size: 32                           # 批次大小
  max_length: 512                          # 最大序列长度
EOF
}

# ==================== 主执行逻辑 ====================
echo "开始deepseek模型实验..."

for dataset in "${DATASETS[@]}"; do
          echo "运行: $dataset"
          # 生成临时配置文件并运行
          config_file="./deepseek_${dataset}.yaml"
          generate_config "$dataset" "$OUTPUT_DIR" "$TYPE" "$CHECKPOINT_PATH" "$PRE_DIS"> "$config_file"
          sbatch_script="./sbatch_deepseek_${dataset}.sh"
          cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH -J deepseek_${dataset}
#SBATCH -o ./log/deepseek_${dataset}_%j.out
#SBATCH -e ./log/%j_stderr_${dataset}
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
source /home/fit/renju/WORK/miniconda3/envs/lxm_infer/bin/activate
conda activate lxm_infer
python $PYTHON_SCRIPT $config_file
rm -f $config_file
EOF
          sbatch "$sbatch_script"
          rm -f "$sbatch_script"
done

echo "实验完成！"