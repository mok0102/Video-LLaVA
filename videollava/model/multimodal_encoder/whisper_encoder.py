import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from transformers import WhisperConfig, WhisperModel, WhisperProcessor


class WhisperModelTower(nn.Module):
    def __init__(self, speech_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.speech_tower_name = speech_tower
        self.select_layer = args.mm_speech_select_layer ## -2
        self.select_feature = getattr(args, 'mm_speech_select_feature', 'cls_patch')
        
        if not delay_load:
            self.load_model()
            
        else:
            self.cfg_only = WhisperConfig.from_pretrained("openai/whisper-large-v2")

    def load_model(self):        
        self.speech_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        self.speech_tower = WhisperModel.from_pretrained("openai/whisper-large-v2") # whisperforconditionalgeneration 써야하나??
        self.speech_tower.requires_grad_(False)
        
        self.is_loaded = True

    def feature_select(self, speech_forward_outs):
        # speech_features = speech_forward_outs.hidden_states[self.select_layer]
        speech_features = speech_forward_outs.last_hidden_state
        
        if self.select_feature == 'patch':
            speech_features = speech_features[:, 1:]
        elif self.select_feature == 'cls_patch': #### always this for speech feature select (for now)
            speech_features = speech_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return speech_features

    @torch.no_grad()
    def forward(self, speeches):
        if type(speeches) is list:
            speech_features = []
            for speech in speeches:
                decoder_input_ids = torch.tensor([[1, 1]]*speech.shape[0]) * self.speech_tower.config.decoder_start_token_id
                speech_forward_out = self.speech_tower(speech.unsqueeze(0).to(device=self.device), decoder_input_ids=decoder_input_ids.to(device=self.device))
                speech_feature = self.feature_select(speech_forward_out).to(speech.dtype)
                speech_features.append(speech_feature)
        else:
            decoder_input_ids = torch.tensor([[1, 1]]*speeches.shape[0]) * self.speech_tower.config.decoder_start_token_id
            speech_forward_outs = self.speech_tower(speeches.to(device=self.device), decoder_input_ids=decoder_input_ids.to(device=self.device))
            speech_features = self.feature_select(speech_forward_outs).to(speeches.dtype)

        return speech_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.speech_tower.dtype

    @property
    def device(self):
        return self.speech_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.speech_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # return self.config.hidden_size
        return 1280

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
