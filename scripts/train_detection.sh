#!/bin/bash

# Tuỳ chỉnh tham số ở đây
EPOCHS=100
BATCH_SIZE=10
LR=1e-4

echo "🚀 Training with epochs=$EPOCHS, batch_size=$BATCH_SIZE, lr=$LR"

python /home/tiennv/phucth/medical/mae/model_detection.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR
