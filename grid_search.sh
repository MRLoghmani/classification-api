#!/bin/bash 

CLASSES=10
PROJECT="complex-benchmark-cifar10"

for BATCH_SIZE in 32 64 128; do
    for LR in 0.3 0.1 0.03; do 
        for MODEL_NAME in "ResNet32CIFAR" ; do 
            for FRAMEWORK in "upstride_type1"; do 
                for RUN in "1"; do
                    if [[ "$FRAMEWORK" == "upstride_type1" ]]; then 
                        FACTOR=2
                    else
                        FACTOR=1
                    fi
                    python train.py --model_name $MODEL_NAME \
                        --input_size 32 32 3 \
                        --num_epochs 1000 \
                        --early_stopping 40 \
                        --factor $FACTOR \
                        --num_classes $CLASSES \
                        --framework $FRAMEWORK \
                        --log_dir "log/${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_LR_${LR}_BS_${BATCH_SIZE}_RUN_${RUN}" \
                        --checkpoint_dir "checkpoints/${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_LR_${LR}_BS_${BATCH_SIZE}_RUN_${RUN}" \
                        --dataloader.name cifar10 \
                        --dataloader.train_list RandomHorizontalFlip Translate Cutout Normalize \
                        --dataloader.val_list Normalize \
                        --dataloader.val_split_id test \
                        --dataloader.Resize.size 36 36 \
                        --dataloader.RandomCrop.size 32 32 3 \
                        --dataloader.Translate.width_shift_range 0.25 \
                        --dataloader.Translate.height_shift_range 0.25 \
                        --dataloader.Cutout.length 4 \
                        --dataloader.batch_size $BATCH_SIZE \
                        --optimizer.lr $LR \
                        --optimizer.lr_decay_strategy.lr_params.patience 20 \
                        --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
                        --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
                         > "${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_LR_${LR}_BS_${BATCH_SIZE}.log" 
                done
            done
        done
    done
done