ulimit -n 4096
set -x

swapon --show
free -h
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8

EXP_DIR=exps/vcoco_l
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port 29179 \
        --use_env \
        main.py \
        --pretrained params/detr-r101-pre-2branch-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet101 \
        --batch_size 2 \
        --num_workers 2 \
        --num_queries 64 \
        --dec_layers 6 \
        --epochs 90 \
        --lr_drop 30 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --gamma_neg 4 \
        --gamma_pos 0 \
        --n_layer 3 \
        --clip_embed_dim 512 \
        --lr 5e-5 \
        --lr_backbone 5e-6 \
        --lr_clip 5e-6 \


