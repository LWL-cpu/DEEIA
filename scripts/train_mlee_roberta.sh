# Learning rate 固定
LR=2e-5

# 定义种子数组
SEEDS=(22 42 66 99 111 1234)

# 遍历种子数组
for SEED in "${SEEDS[@]}"
do
    # 设置工作路径
    work_path=exp/mlee/$SEED
    mkdir -p $work_path

    # 运行 Python 脚本
    python -u engine.py \
        --dataset_type=MLEE \
        --context_representation=decoder \
        --model_name_or_path=roberta-large \
        --role_path=./data/MLEE/MLEE_role_name_mapping.json \
        --prompt_path=./data/prompts/prompts_MLEE_full.csv \
        --seed=$SEED \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --batch_size=4 \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_dec_seq_length 512 \
        --window_size 250 \
        --bipartite \
        --lamb 0.1
done
