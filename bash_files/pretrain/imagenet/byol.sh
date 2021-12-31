python3 ../../../main_pretrain.py \
    --dataset imagenet \
    --backbone resnet50 \
    --data_dir ~/workspace/datasets/ \
    --train_dir imagenet/train \
    --val_dir imagenet/val \
    --max_epochs 100 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 4.0 \
    --classifier_lr 1.0 \
    --accumulate_grad_batches 16 \
    --weight_decay 1e-6 \
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
    --project Imagenet1K-100ep \
    --entity kaistaim \
    --wandb \
    --save_checkpoint \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --lam 0.1 \
    --tau_decor 0.1 \
    --our_loss True \
    --keep_previous_checkpoints \
    --checkpoint_frequency 5 \

# python3 ../../../main_pretrain.py \
#     --dataset imagenet \
#     --backbone resnet50 \
#     --data_dir ~/workspace/datasets/ \
#     --train_dir imagenet/train \
#     --val_dir imagenet/val \
#     --max_epochs 100 \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --accelerator gpu \
#     --strategy ddp \
#     --sync_batchnorm \
#     --precision 16 \
#     --optimizer sgd \
#     --lars \
#     --eta_lars 0.001 \
#     --exclude_bias_n_norm \
#     --scheduler warmup_cosine \
#     --lr 0.45 \
#     --classifier_lr 0.2 \
#     --accumulate_grad_batches 16 \
#     --weight_decay 1e-6 \
#     --batch_size 64 \
#     --num_workers 4 \
#     --brightness 0.4 \
#     --contrast 0.4 \
#     --saturation 0.2 \
#     --hue 0.1 \
#     --gaussian_prob 1.0 0.1 \
#     --solarization_prob 0.0 0.2 \
#     --num_crops_per_aug 1 1 \
#     --name byol_res50_8gpu_bsz64 \
#     --project Imagenet1K-200ep \
#     --entity kaistaim \
#     --wandb \
#     --save_checkpoint \
#     --method byol \
#     --proj_output_dim 256 \
#     --proj_hidden_dim 4096 \
#     --pred_hidden_dim 4096 \
#     --base_tau_momentum 0.99 \
#     --final_tau_momentum 1.0 \
#     --lam 0.1 \
#     --tau_decor 0.1 \
#     --our_loss False \
#     --keep_previous_checkpoints \
#     --checkpoint_frequency 5 \
