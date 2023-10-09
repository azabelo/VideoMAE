OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
DATA_PATH='pretrain.csv'

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 \
          --master_port 21416 --nnodes=1 \
          --node_rank=0 --master_addr="localhost" \
          run_mae_pretraining.py \
          --data_path ${DATA_PATH} \
          --mask_type tube \
          --mask_ratio 0.9 \
          --model pretrain_videomae_base_patch16_224 \
          --decoder_depth 4 \
          --batch_size 8 \
          --num_frames 16 \
          --sampling_rate 4 \
          --opt adamw \
          --opt_betas 0.9 0.95 \
          --warmup_epochs 40 \
          --save_ckpt_freq 300 \
          --epochs 4801 \
          --log_dir ${OUTPUT_DIR} \
          --output_dir ${OUTPUT_DIR}