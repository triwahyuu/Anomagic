"""
Hyperparameter sweep on the real KDW inputs (run_infer_real_kdw.py's baseline was scale=1.0,
strength=0.6 and produced only a weak, localized edit). Tries stronger scale/strength values on
the SAME real reference + targets + masks, to see if a more visible defect is achievable.
Every output gets its own distinctly-named file (never overwrites the baseline or any other
sweep result) so the progression across settings can be traced afterward.
"""
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kdw")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "real_kdw", "sweep")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"

CASES = [
    {"name": "sample1_202605", "target_image": "sample1_target_512.png", "target_mask": "sample1_target_mask_512.png"},
    {"name": "sample2_202609", "target_image": "sample2_target_512.png", "target_mask": "sample2_target_mask_512.png"},
]

SWEEP = [
    (2.0, 0.6),
    (4.0, 0.6),
    (1.0, 0.8),
    (2.0, 0.8),
    (4.0, 0.9),
]


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    return Image.open(path).convert("L")


def main():
    from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from ip_adapter.ip_adapter_anomagic import Anomagic

    base_hf = "base_hf"
    vae_hf = "vae_hf"
    image_encoder_path = "ip_adapter_hf/models/image_encoder"

    print("Building shared pipeline components...")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_hf, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(base_hf, "text_encoder")).to(DEVICE, dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(vae_hf).to(DEVICE, dtype=torch.float16)
    unet_config = UNet2DConditionModel.load_config(os.path.join(base_hf, "unet"))
    unet = UNet2DConditionModel.from_config(unet_config).to(DEVICE, dtype=torch.float16)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False, steps_offset=1,
    )
    pipe = StableDiffusionInpaintPipelineLegacy(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=noise_scheduler,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    model = Anomagic(pipe, image_encoder_path, "checkpoint/anomagic.bin", "checkpoint/attention_module.bin", DEVICE)
    print("Shared components ready.\n")

    ref_image = load_rgb(os.path.join(REAL_DATA, "crack2_ref_512.png"))
    ref_mask = load_mask(os.path.join(REAL_DATA, "crack2_ref_mask_512.png"))

    results = []
    seed_ctr = 300
    for case in CASES:
        target_image = load_rgb(os.path.join(REAL_DATA, case["target_image"]))
        target_mask = load_mask(os.path.join(REAL_DATA, case["target_mask"]))

        for scale, strength in SWEEP:
            seed_ctr += 1
            tag = f"scale{scale}_strength{strength}"
            print(f"--- {case['name']} {tag} ---")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            torch.manual_seed(seed_ctr)
            images = model.generate(
                pil_image=ref_image,
                mask_image_0=ref_mask,
                image=target_image,
                mask_image=target_mask,
                prompt="a photo of a pipe interior with a crack",
                num_samples=1,
                num_inference_steps=20,
                strength=strength,
                scale=scale,
                guidance_scale=7.5,
                seed=seed_ctr,
            )
            gen_time = time.time() - t0
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9

            if images:
                out_path = os.path.join(OUT_DIR, f"{case['name']}_{tag}.png")
                images[0].save(out_path)
                status = "OK"
            else:
                status = "NO IMAGE"
            print(f"  status={status} time={gen_time:.2f}s peak_alloc={peak_alloc:.3f}GB")
            results.append({"case": case["name"], "scale": scale, "strength": strength,
                             "status": status, "time_s": round(gen_time, 2)})

    print("\n" + "=" * 60)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
