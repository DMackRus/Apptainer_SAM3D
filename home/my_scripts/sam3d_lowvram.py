"""
Low-VRAM mode for SAM 3D Objects (for 16 GB GPUs such as the RTX 4080 SUPER).

Upstream keeps every model on the GPU at once (~12.8 GB of weights), which leaves
no room for activations on a 16 GB card. The pipeline runs in stages though:

    depth (MoGe)  ->  sparse structure (ss_generator + DINO + ss_decoder)
                  ->  structured latent (slat_generator + DINO)  ->  decoders

so this keeps all models in CPU RAM and moves only the current stage's models onto
the GPU (peak ~6.2 GB of weights). Costs a few seconds of CPU<->GPU copies per run.
Upstream code is not modified, the pipeline methods are wrapped at runtime.

Usage:
    sys.path.append("/scripts")
    from sam3d_lowvram import enable_low_vram
    inference = Inference(config_path, compile=False)
    enable_low_vram(inference)
"""
import functools

import torch


def enable_low_vram(inference, verbose=True):
    p = inference._pipeline
    m = p.models
    emb = p.condition_embedders

    stages = {
        "compute_pointmap": [p.depth_model.model],
        "sample_sparse_structure": [m["ss_generator"], emb["ss_condition_embedder"], m["ss_decoder"]],
        "sample_slat": [m["slat_generator"], emb["slat_condition_embedder"]],
        "decode_slat": [m["slat_decoder_mesh"], m["slat_decoder_gs"], m["slat_decoder_gs_4"]],
    }
    all_models = {id(x): x for group in stages.values() for x in group if x is not None}
    on_gpu = set()

    def load(group, name):
        wanted = {id(x) for x in group if x is not None}
        for i in list(on_gpu - wanted):
            all_models[i].to("cpu")
            on_gpu.discard(i)
        torch.cuda.empty_cache()
        for i in wanted - on_gpu:
            all_models[i].to(p.device)
            on_gpu.add(i)
        if verbose:
            print(f"[low-vram] {name}: {torch.cuda.memory_allocated() / 2**30:.1f} GB on GPU", flush=True)

    for name, group in stages.items():
        original = getattr(p, name)

        @functools.wraps(original)
        def wrapped(*args, _orig=original, _group=group, _name=name, **kwargs):
            load(_group, _name)
            return _orig(*args, **kwargs)

        setattr(p, name, wrapped)

    # Start with everything in CPU RAM
    for x in all_models.values():
        x.to("cpu")
    torch.cuda.empty_cache()
    if verbose:
        print(f"[low-vram] models moved to CPU, {torch.cuda.memory_allocated() / 2**30:.1f} GB left on GPU", flush=True)
    return inference
