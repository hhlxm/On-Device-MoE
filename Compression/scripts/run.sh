#!/bin/bash
###
 # @Author: hhlxm 578723542@qq.com
 # @Date: 2025-05-14 02:40:47
 # @LastEditors: hhlxm 578723542@qq.com
 # @LastEditTime: 2025-05-15 14:24:30
 # @FilePath: /lxm/Compression/scripts/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
#SBATCH --partition=a01
#SBATCH --gres=gpu:1
#SBATCH --job-name=compression
#SBATCH --output=/home/fit/renju/WORK/lxm/Compression/results/log_all3bit_5shot.out

python /home/fit/renju/WORK/lxm/Compression/eval.py \
    --model_id /home/fit/renju/WORK/lxm/Compression/models/DeepSeek_V2_Lite-all3bit \
     --output_file /home/fit/renju/WORK/lxm/Compression/results/DeepSeek_V2_Lite-ex_all3bit.json

# python /home/fit/renju/WORK/lxm/Compression/quantization.py  --bits 3
