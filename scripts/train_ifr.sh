#! /bin/bash
# bash scripts/train_diffcap.sh CONFIG GPU

while true; do
    PORT=$((20000 + RANDOM % 101))
    if ! ss -tuln | grep -q ":$PORT"; then
        export MASTER_PORT=$PORT
        break
    fi
done

config=$1
GPU=$2
gpunum=$(awk -F',' '{print NF}' <<< "$GPU")
echo "Using $gpunum GPUs"
torchrun --nproc_per_node $gpunum --master_port=$MASTER_PORT ./train_ifr.py --config $config --map