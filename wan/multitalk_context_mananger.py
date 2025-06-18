import types
from contextlib import contextmanager
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from .distributed.xdit_context_parallel import (
    usp_dit_forward_multitalk,
    usp_attn_forward_multitalk,
    usp_crossattn_multi_forward_multitalk
)


from torch import distributed as dist

@contextmanager
def parallel_context(model, use_usp, ulysses_size, ring_size, para_batch_size):
    original_attn_forwards = []
    original_crossattn_forwards = []
    original_model_forward = model.forward

    if ulysses_size > 1 or ring_size > 1 or para_batch_size > 1:
        world_size=dist.get_world_size()
        assert ulysses_size * ring_size * para_batch_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        assert para_batch_size == 1 or para_batch_size == 3, f"The para_batch_size should be 1 or 3, but got {para_batch_size}."

        rank = dist.get_rank()
        init_distributed_environment(
            rank=rank, 
            world_size=world_size
        )

        initialize_model_parallel(
            classifier_free_guidance_degree=para_batch_size,
            sequence_parallel_degree = ulysses_size * ring_size,
            ring_degree=ring_size,
            ulysses_degree=ulysses_size,
        )

        use_usp = ulysses_size * ring_size > 1
        if use_usp:
            for block in model.blocks:
                original_attn_forwards.append(block.self_attn.forward)
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn
                )
                original_crossattn_forwards.append(block.audio_cross_attn.forward)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn
                )

            model.forward = types.MethodType(usp_dit_forward_multitalk, model)
        sp_size = get_sequence_parallel_world_size()
        print("Using sequence parallel size:", sp_size)
    else:
        sp_size = 1

    try:
        yield sp_size
    finally:
        if use_usp:
            # Restore original attention forwards
            for block, original_forward in zip(model.blocks, original_attn_forwards):
                block.self_attn.forward = original_forward
            for block, original_forward in zip(model.blocks, original_crossattn_forwards):
                block.audio_cross_attn.forward = original_forward

            # Restore model forward and original sp flag
            model.forward = original_model_forward
