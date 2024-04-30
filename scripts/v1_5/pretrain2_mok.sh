
JSON_FOLDER="/home/mok/module/Video-LLaVA/llava_all_image_video/ft_json"
IMAGE_FOLDER="/home/mok/module/Video-LLaVA/llava_all_image_video"
VIDEO_FOLDER="/node_data/hyun/mok/data/MELD.Raw/train_splits"
SPEECH_FOLDER="/node_data/hyun/mok/data/MELD.Raw/train_splits"

cd /home/mok/module/Video-LLaVA-Speech
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed  --include localhost:0,1  videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/meld_train_debug.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --tune_mm_sp_mlp_adapter True \
    --pretrain_mm_mlp_adapter /home/mok/module/Video-LLaVA/checkpoints/videollava-7b-pretrain/mm_projector.bin \
    --freeze_backbone True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_spch_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints/videollava-7b-aud-debug  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "/node_data/mok/cache/hub"
