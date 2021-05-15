#!/bin/bash
#SBATCH -p gpu3          # CPU 核心数
#SBATCH -c 8
#SBATCH --mem 48000     # 内存（MB）
#SBATCH --gres gpu:4  # 分配1个GPU（纯CPU任务不用写）
#SBATCH -N 1
python pretrain.py
  