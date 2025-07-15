CUDA_VISIBLE_DEVICES=1 python train.py \
        --batch_size 196 \
        --epochs 10 \
        --model mae_vit_base_patch16 \
        --input_size 224 \
        --weight_decay 0.05 \
        --blr 1.5e-4 \
        --warmup_epochs 4 \
        --dataset_name_train "inpaint-context/train-mae-update-furniture" \
        --image_folder \
            "/mnt/Datadrive/tiennv/data/final" \
            "/mnt/Datadrive/datasets/ade20k/ade20k" \
            "/mnt/Datadrive/datasets/ade20k/pascal-context" \
            "/mnt/Datadrive/datasets/coco2017/train" \
            "/mnt/Datadrive/datasets/coco2017/val" \
        --do_train \
        --do_eval \
        --mask_ratio 0.5 \
        --mask_min 0 \
        --mask_max 1 \
        --cache_dir .cache \
        --output_dir outputs_rand_4_bitwise_3_semi_objmask_300/files \
        --log_dir outputs_rand_4_bitwise_3_semi_objmask_300/logs \
        --weights checkpoints/mae_visualize_vit_base/rand_4_bitwise_3_semi_objmask.pth \
        --mask_mode 'rand' 'bitwise' 'semi_objmask' \
        --dis_mask 0.4 0.3 0.3

# torch.distributed.launch