#!/bin/bash
# Docker related
image="docker.v2.aispeech.com/hpc/ai_on_device_base-atec:0.0.2"
export cpu_cmd="vc submit --image $image --partition pdcpu --mem-per-task 6G --cpu-per-task 1 --sync"
export cpu_cmd_big="vc submit --image $image --partition pdcpu --mem-per-task 24G --cpu-per-task 2 --sync"
export cpu_cmd_small="vc submit --image $image --partition pdcpu --mem-per-task 6G --cpu-per-task 1 --sync"
export gpu_cmd="vc submit --image $image --partition pdgpu-a10 --gpu-per-task 1 --mem-per-task 6G --cpu-per-task 3 --sync"
export local_cmd="nohup /hpc_stor01/home/jifa.cai/tools/miniconda3/envs/py37/bin"

# image initial env
export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH=

# Kaldi
TOOL_DIR=/hpc_stor01/project/ezdl
export PATH=$TOOL_DIR/kaldi_cpu/bin:$PATH
export LD_LIBRARY_PATH=$TOOL_DIR/kaldi_cpu/lib:$LD_LIBRARY_PATH

# Python packages
MYDIR=/hpc_stor01/group/on_device/gtools
export PYTHONPATH=$MYDIR/pytorch-asr:$PYTHONPATH

