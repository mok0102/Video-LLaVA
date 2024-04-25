import torch
from videollava.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_SPEECH_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, tokenizer_image_speech_token, get_model_name_from_path, KeywordsStoppingCriteria

import pdb

import whisper

def main():
    disable_torch_init()
    video = '/home/mok/module/Video-LLaVA/test.mp4'
    
    ### laugh reasoing 가능, scene에서 relation 추론 가능, emotion 추론 가능
    inp = 'Give me the reason why the person laughed in this clip. \
I also give you transcription and the relation between two people.\
Utterance: Why do you love me? Oh, I love you because youre my firstborn. Youre my game changer, my life changer.\
Suck it, siblings. '
    model_base = "lmsys/vicuna-7b-v1.5"
    # model_path = "./checkpoints/videollava-7b-lora"
    # model_path = "/home/mok/module/Video-LLaVA/checkpoints/videollava-7b"
    model_path = "/home/mok/module/Video-LLaVA-aud/checkpoints/videollava-7b-aud-debug"
    cache_dir = '/home/mok/module/Video-LLaVA/cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, model_name)#, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    speech_processor = processor['speech']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
        
    # import pdb; pdb.set_trace()
    
    audio = whisper.load_audio(video)
    # audio = whisper.pad_or_trim(audio)
    
    speech_tensor = [speech_processor(audio, return_tensors='pt').input_features]
    
    if type(speech_tensor) is list:
        speech = [speech.to(model.device, dtype=torch.float16) for speech in speech_tensor]
        
    else:
        speech = speech_tensor.to(model.device, dtype=torch.float16)
        
    
        

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    # inp = ' '.join([DEFAULT_IMAGE_TOKEN] * 1) + '\n' + inp
    inp = ' '.join([DEFAULT_SPEECH_TOKEN] * 1) + '\n' + inp ### speech가 맨앞에들어가!!!
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = tokenizer_image_speech_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            speeches=speech,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
        

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

if __name__ == '__main__':
    main()