from PIL import Image

from huggingface_hub import snapshot_download
from .wan.configs import WAN_CONFIGS
from .wan import MultiTalkPipeline

from transformers import Wav2Vec2FeatureExtractor
from .src.audio_analysis.wav2vec2 import Wav2Vec2Model

from .wan.multitalk_context_mananger import parallel_context
from .wan.utils.multitalk_utils import save_video_ffmpeg

from .generate_multitalk import (
    get_embedding,
    audio_prepare_single,
    audio_prepare_multi,
)
def create_pipeline(
        multitalk_model = "Runware/multitalk-14B-480P",
        wav2vec2_model = "TencentGameMate/chinese-wav2vec2-base",
        device = "cuda",
        cache_dir = None
):
    cfg = WAN_CONFIGS['multitalk-14B']
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec2_model, cache_dir = cache_dir).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_model, cache_dir = cache_dir)

    if multitalk_model.startswith("/"):
        # If the model path is a local path, we don't need to download it
        multitalk_model_path = multitalk_model
    else:
        multitalk_model_path = snapshot_download(
            multitalk_model,
            cache_dir=cache_dir
        )

    pipeline =  MultiTalkPipeline(
        config=cfg,
        checkpoint_dir=multitalk_model_path,
    )

    return audio_feature_extractor, audio_encoder, pipeline


## This function is re-written from generate_multitalk.py
## Mostly in order to prevent file I/O operations
def prepare_input_data(
    prompt : str ,
    cond_image : Image.Image | str,
    cond_audio : str | list[str],
    audio_feature_extractor : Wav2Vec2FeatureExtractor,
    audio_encoder : Wav2Vec2Model,
    audio_type : str, ## "para" or "add"
    boxes : list[list[int]] | None = None,  ## If None, it will split the image into two equal halves
    sample_rate = 16000
):
    if isinstance(cond_image, str):
        cond_image = Image.open(cond_image).convert("RGB")

    if not isinstance(cond_audio, list):
        cond_audio = [cond_audio]

    num_audio = len(cond_audio)
    if num_audio == 1:
        human_speech = audio_prepare_single(cond_audio[0], sample_rate=sample_rate)
        audio_embeddings_1 = get_embedding(human_speech, audio_feature_extractor, audio_encoder, device="cuda")
        audio_embeddings_2 = None
        video_audio = human_speech
    elif num_audio == 2:
        left_path = cond_audio[0]
        right_path = cond_audio[1]
        human_speech1, human_speech_2, sum_human_speech = audio_prepare_multi(
            left_path=left_path,
            right_path=right_path,
            sample_rate=sample_rate,
            audio_type=audio_type,
        )
        audio_embeddings_1 = get_embedding(human_speech1, audio_feature_extractor, audio_encoder, device="cuda")
        audio_embeddings_2 = get_embedding(human_speech_2, audio_feature_extractor, audio_encoder, device="cuda")
        video_audio = sum_human_speech
    else:
        raise ValueError("cond_audio should be a list with 1 or 2 audio files.")

    input_data = {
        "prompt": prompt,
        "cond_image": cond_image,
        "cond_audio" : {
            "person1": audio_embeddings_1,
        },
        "video_audio": video_audio,
    }
    if audio_embeddings_2 is not None:
        input_data["cond_audio"]["person2"] = audio_embeddings_2
    if boxes is not None:
        if len(boxes) != num_audio:
            raise ValueError(f"Number of boxes {len(boxes)} does not match number of audios {num_audio}.")
        input_data["boxes"] = boxes
    return input_data


def test_helper(use_usp = True, ulysses_size=1, ring_size=1, para_batch_size=1):
    audio_feature_extractor, audio_encoder, pipeline = create_pipeline(
        device="cuda"
    )
    input_data = prepare_input_data(
        prompt = "In a cozy recording studio, a man and a woman are singing together. The man, with tousled brown hair, stands to the left, wearing a light green button-down shirt. His gaze is directed towards the woman, who is smiling warmly. She, with wavy dark hair, is dressed in a black floral dress and stands to the right, her eyes closed in enjoyment. Between them is a professional microphone, capturing their harmonious voices. The background features wooden panels and various audio equipment, creating an intimate and focused atmosphere. The lighting is soft and warm, highlighting their expressions and the intimate setting. A medium shot captures their interaction closely.",
        cond_image="examples/multi/3/multi3.png",
        cond_audio=[
            "examples/multi/3/1-man.WAV",
            "examples/multi/3/1-woman.WAV"
        ],
        audio_feature_extractor = audio_feature_extractor,
        audio_encoder = audio_encoder,
        audio_type="para",
    )

    model_scale = 'multitalk-480'
    use_teacache = True
    teacache_thresh = 0.5

    use_apg = True
    apg_momentum = -0.75
    apg_norm_threshold = 55

    from types import SimpleNamespace

    extra_args = SimpleNamespace()
    extra_args.use_teacache = use_teacache
    extra_args.teacache_thresh = teacache_thresh
    extra_args.size = model_scale
    extra_args.use_apg = use_apg
    extra_args.apg_momentum = apg_momentum
    extra_args.apg_norm_threshold = apg_norm_threshold

    for i in range(2):
        with parallel_context(pipeline.model, use_usp=use_usp, ulysses_size=ulysses_size, ring_size=ring_size, para_batch_size=para_batch_size) as sp_size:
            video = pipeline.generate(
                input_data,
                size_buckget=model_scale,
                motion_frame=25,
                frame_num=81,
                shift=5.0,
                sampling_steps=10,
                text_guide_scale=5.0,
                audio_guide_scale=4.0,
                n_prompt="",
                seed=-1,
                offload_model=True,
                max_frames_num=1000,
                face_scale=0.05,
                progress=True,
                batched_cfg=True,
                extra_args=extra_args
            )

        save_video_ffmpeg(video, f"test_{i}.mp4", [input_data['video_audio']])




if __name__ == "__main__":
    import os
    import torch
    import torch.distributed as dist


    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    test_helper(use_usp=True, ulysses_size=1, ring_size=1, para_batch_size=1)
    # test(use_usp=False)  # Uncomment to test without USP
