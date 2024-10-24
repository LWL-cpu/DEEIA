# Learning rate固定
LR=2e-5

# 定义种子数组
SEEDS=(22 42 66 99 111 1234)

# 遍历种子数组
for SEED in "${SEEDS[@]}"
do
    # 设置工作路径
    work_path=exps/rams_large/$SEED/$LR
    mkdir -p $work_path

    # 运行Python脚本
    python -u engine.py \
        --model_type=DEEIA \
        --dataset_type=rams \
        --model_name_or_path=roberta-large \
        --role_path=./data/dset_meta/description_rams.csv \
        --prompt_path=./data/prompts/prompts_rams_full.csv \
        --seed=$SEED \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_prompt_seq_length 330 \
        --lamb  1e-5 \
        --bipartite

done
