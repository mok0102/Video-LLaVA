
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_video_tower, build_speech_tower
from .multimodal_projector.builder import build_vision_projector, build_speech_projector

from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_PATCH_TOKEN
# remove the padding using attention_mask -- TODO: double check
input_ids = [torch.rand(2094)]
labels = [torch.rand(2094)]

new_input_embeds = []
new_labels = []
cur_image_idx = 0
cur_speech_idx = 0

for batch_idx, cur_input_ids in enumerate(input_ids):
    num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    num_speeches = (cur_input_ids == SPEECH_TOKEN_INDEX).sum() # should be 1
    # print(num_images, cur_input_ids)
    if num_images+num_speeches == 0:
        # cur_image_features = image_features[cur_image_idx]
        cur_image_features = torch.rand(300, 4096)
        cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
        if num_speeches == 0:
            cur_speech_features = torch.rand(2, 4096)
        cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_speech_features[0:0]], dim=0)
        new_input_embeds.append(cur_input_embeds)
        new_labels.append(labels[batch_idx])
        cur_image_idx += 1
        cur_speech_idx += 1
        continue
    
    image_token_indices = [-1] + torch.where((cur_input_ids == IMAGE_TOKEN_INDEX)|cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    cur_input_ids_noim = []
    cur_labels = labels[batch_idx]
    cur_labels_noim = []
    for i in range(len(image_token_indices) - 1):
        cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
        cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    split_sizes = [x.shape[0] for x in cur_labels_noim]
    cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    cur_new_input_embeds = []
    cur_new_labels = []
    print(num_images)

    for i in range(num_images + num_speeches + 1): # should be num_images+2
        cur_new_input_embeds.append(cur_input_embeds_no_im[i])
        cur_new_labels.append(cur_labels_noim[i])
        if i < num_images:
            # print(cur_image_idx)
            if i==0:
                cur_image_features = speech_features[cur_speech_idx]
                speech_idx+=1
                
            else:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
            cur_new_input_embeds.append(cur_image_features)
            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    cur_new_labels = torch.cat(cur_new_labels)
    print('llava arch cur_new_labels', cur_new_labels)
    print(torch.all(cur_new_labels == -100))