gpu=$1
type=$2
port=$3
echo type=$type
echo GPU=$gpu
echo port=$port

export CUDA_VISIBLE_DEVICES=$gpu

gpu_num=`expr ${#gpu} / 2 + 1`
echo gpu_num:${gpu_num}

if [ ${#gpu} -gt 2 ]
then
    echo "Using Multiple GPU:${gpu}"
    if [ ${#3} -eq 0 ]
    then
        port=12345
    else
        port=$port
    fi
    echo nproc_per_node=${gpu_num}

    python3 -m torch.distributed.launch \
    --master_addr=127.0.0.1 \
    --master_port=$port \
    --nproc_per_node=${gpu_num} \
    train.py\
    --cfg $type.yaml \
    --tb_log tb_log/tb_log_$type

else
    echo "Using Single GPU:${gpu}"
    python3 train.py \
    --cfg $type.yaml \
    --tb_log tb_log/tb_log_$type
fi