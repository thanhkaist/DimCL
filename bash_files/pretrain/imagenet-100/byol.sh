python3 ../../../main_pretrain.py \
    --dataset imagenet100 \
    --backbone resnet50 \
    --data_dir ~/workspace/datasets/ \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --max_epochs 200 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 4.0 \
    --classifier_lr 1.0 \
    --weight_decay 1e-5 \
    --batch_size 64 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name byol_res50_ours_8gpu_bsz64 \
    --entity kaistaim \
    --project Imagenet100-200ep \
    --wandb \
    --save_checkpoint \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 8192 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --lam 0.1 \
    --tau_decor 0.1 \
    --our_loss True \
    --knn_eval \
    # --dali \

# python3 ../../../main_pretrain.py \
#     --dataset imagenet100 \
#     --backbone resnet18 \
#     --data_dir ~/workspace/datasets/ \
#     --train_dir imagenet-100/train \
#     --val_dir imagenet-100/val \
#     --max_epochs 200 \
#     --gpus 0 \
#     --accelerator gpu \
#     --strategy ddp \
#     --sync_batchnorm \
#     --precision 16 \
#     --optimizer sgd \
#     --lars \
#     --grad_clip_lars \
#     --eta_lars 0.02 \
#     --exclude_bias_n_norm \
#     --scheduler warmup_cosine \
#     --lr 0.5 \
#     --classifier_lr 0.1 \
#     --weight_decay 1e-5 \
#     --batch_size 256 \
#     --num_workers 4 \
#     --brightness 0.4 \
#     --contrast 0.4 \
#     --saturation 0.2 \
#     --hue 0.1 \
#     --gaussian_prob 1.0 0.1 \
#     --solarization_prob 0.0 0.2 \
#     --num_crops_per_aug 1 1 \
#     --name byol_res18 \
#     --entity kaistaim \
#     --project Imagenet100-200ep \
#     --wandb \
#     --save_checkpoint \
#     --method byol \
#     --proj_output_dim 256 \
#     --proj_hidden_dim 4096 \
#     --pred_hidden_dim 8192 \
#     --base_tau_momentum 0.99 \
#     --final_tau_momentum 1.0 \
#     --knn_eval \
#     --lam 0.1 \
#     --tau_decor 0.1 \
#     --our_loss False \
