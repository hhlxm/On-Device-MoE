# 定义数据集和对应的 few-shot 策略
###
 # @Author: hhlxm 578723542@qq.com
 # @Date: 2025-04-16 01:41:13
 # @LastEditors: hhlxm 578723542@qq.com
 # @LastEditTime: 2025-05-04 17:15:56
 # @FilePath: /lxm/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
declare -A dataset_shots
dataset_shots=(
    # ["mmlu"]=5
    # ["gsm8k"]=5
    ["hellaswag"]=10
    # ["humaneval"]=0
)

pre_dis=0
pre_ahead=3

# 模型和检查点路径
model_path="/home/fit/renju/WORK/lxm/models/Phi_3_5_MoE_instruct"
# 提取模型名
model_name=$(basename ${model_path})
checkpoint_base="null"
# checkpoint_base="/home/fit/renju/WORK/lxm/Predict/results/DeepSeek_V2_Lite/output_pregate_finetune_dis26/deepseek_v2_lite/alpaca/alpaca/checkpoints/checkpoint_step_3100.pt"

# 循环遍历数据集和对应的 shot 策略
for dataset in "${!dataset_shots[@]}"; do
    shot=${dataset_shots[$dataset]}
    
    # 动态生成输出文件路径
    output_file="./results/results_${dataset}_${shot}shot"
    checkpoint_path="${checkpoint_base}"

    # 情况1：传入 checkpoint_path
    sbatch_script="sbatch_${dataset}_${shot}_with_checkpoint.sh"
    cat > ${sbatch_script} << EOF
#!/bin/bash
#SBATCH -J eval_${dataset}_${shot}_ckpt
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o ./log/${model_name}_%j_stdout_${dataset}_${shot}shot_dis${pre_dis}_ah${pre_ahead}
#SBATCH -e ./log/%j_stderr_${dataset}_${shot}shot
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
source /home/fit/renju/WORK/miniconda3/envs/lxm_eval/bin/activate
conda activate lxm_eval
python lm_evaluation_harness/data_para.py \
    --model_path ${model_path} \
    --checkpoint_path ${checkpoint_path} \
    --subtasks ${dataset} \
    --num_fewshot ${shot} \
    --batch_size "auto:4" \
    --output_file ${output_file}_with_checkpoint.json \
    --pre_dis ${pre_dis} \
    --pre_ahead ${pre_ahead} 
EOF

    # 提交 sbatch 任务（带 checkpoint）
    sbatch ${sbatch_script}

    rm sbatch_${dataset}_${shot}_*.sh
done