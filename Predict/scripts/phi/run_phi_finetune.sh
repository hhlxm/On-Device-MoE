#!/bin/bash
###
 # @Author: hhlxm 578723542@qq.com
 # @Date: 2025-04-11 13:39:43
 # @LastEditors: hhlxm 578723542@qq.com
 # @LastEditTime: 2025-04-21 01:03:58
 # @FilePath: /lxm/Predict/scripts/run_deepseek_test.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# 配置目录，存储所有 YAML 文件
CONFIG_DIR="/home/fit/renju/WORK/lxm/Predict/config/phi/finetune"

# Python 脚本路径
PYTHON_SCRIPT="/home/fit/renju/WORK/lxm/Predict/finetune.py"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist."
    exit 1
fi

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT does not exist."
    exit 1
fi



# 遍历 CONFIG_DIR 中的所有 .yaml 文件
for config_file in "$CONFIG_DIR"/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "Running $PYTHON_SCRIPT with config: $config_file"
        python "$PYTHON_SCRIPT" "$config_file"
        
        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "Successfully completed: $config_file"
        else
            echo "Error occurred while running: $config_file"
        fi
        echo "----------------------------------------"
    else
        echo "No .yaml files found in $CONFIG_DIR"
        break
    fi
done

echo "All configurations processed."