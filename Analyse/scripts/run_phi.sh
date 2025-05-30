#!/bin/bash

# 配置目录，存储所有 YAML 文件
CONFIG_DIR="/home/fit/renju/WORK/lxm/Analyse/config_phi_temp"

# Python 脚本路径
PYTHON_SCRIPT="/home/fit/renju/WORK/lxm/Analyse/eval.py"

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
        CUDA_VISIBLE_DEVICES=6,7 python "$PYTHON_SCRIPT" "$config_file"
        
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