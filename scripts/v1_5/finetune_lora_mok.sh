

JSON_FOLDER="/home/mok/module/Video-LLaVA/llava_all_image_video/ft_json"
IMAGE_FOLDER="/home/mok/module/Video-LLaVA/llava_all_image_video"
VIDEO_FOLDER="/node_data/hyun/mok/data/MELD.Raw/train_splits"
SPEECH_FOLDER="/node_data/hyun/mok/data/MELD.Raw/train_splits"
cd /home/mok/module/Video-LLaVA-Speech

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed videollava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/meld_train.json \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --speech_folder ${SPEECH_FOLDER} \
    --speech_tower WhisperModel \
    --mm_projector_type mlp2x_gelu \
    --mm_sp_hidden_size 1280 \
    --pretrain_mm_mlp_adapter /node_data/hyun/mok/Video-LLaVA/checkpoints/videollava-7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_speech_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b-aud \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
