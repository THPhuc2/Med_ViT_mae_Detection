CUDA_VISIBLE_DEVICES=0,1 python train_ptln.py \
        --devices 0 1 \
        --batch_size 16 \
        --accum_iter 1 \
        --epochs 150 \
        --model mae_vit_huge_patch14 \
        --input_size 224 \
        --weight_decay 0.05 \
        --blr 1.0e-4 \
        --warmup_epochs 4 \
        --dataset_name_train "THP2903/image_x-ray_8bit_mae_mask" \
        --image_folder \
            "/home/tiennv/phucth/medical/data_mae/data_mae/1" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/2" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/3" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/4" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/5" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/6" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/7" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/8" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/9" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/10" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/11" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/12" \
            "/home/tiennv/phucth/medical/data_mae/data_mae/13" \
        --do_train \
        --do_eval \
        --mask_ratio 0.5 \
        --mask_min 0 \
        --mask_max 1 \
        --cache_dir .cache \
        --output_dir outputs_rand_4_bitwise_3_semi_objmask_150_huge/files \
        --log_dir outputs_rand_4_bitwise_3_semi_objmask_150_huge/logs \
        --weights '/home/tiennv/phucth/medical/mae/checkpoint/mae_pretrain_vit_huge.pth' \
        --mask_mode 'rand' 'bitwise' 'semi_objmask' \
        --dis_mask 0.4 0.3 0.3

# torch.distributed.launch