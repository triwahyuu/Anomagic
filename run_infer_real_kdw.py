"""
Real Anomagic inference on real KDW pipe photos -- the first genuine end-to-end test with
actual KDW data (not MVTec stand-ins). See playground/plan/20260909_02_real_kdw_data_crack_generation_plan.md.

Reference: playground/video/crack/crack2.png, cropped to 512x512 (assets/kdw/crack2_ref_512.png)
           mask hand-annotated by the user (crack2-annotated.png), extracted + verified by overlay
           (assets/kdw/crack2_ref_mask_512.png).

Targets:
  1. playground/video/sample/202605-no1-sample1.png (poor quality, hazy/blurry -- flagged)
  2. playground/video/sample/202609-no1-sample2.png (clear, well-exposed)
  Both cropped to 512x512; target masks are the user's own drawn shape (from
  202609-no2-sample2-annotated.png), repositioned/scaled/mirrored per the user's own direction,
  confirmed by overlay before this run.
"""
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kdw")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "real_kdw")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    return Image.open(path).convert("L")


CASES = [
    {"name": "sample1_202605", "target_image": "sample1_target_512.png", "target_mask": "sample1_target_mask_512.png"},
    {"name": "sample2_202609", "target_image": "sample2_target_512.png", "target_mask": "sample2_target_mask_512.png"},
]


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
    for i, case in enumerate(CASES):
        print(f"--- Case {case['name']} ---")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        target_image = load_rgb(os.path.join(REAL_DATA, case["target_image"]))
        target_mask = load_mask(os.path.join(REAL_DATA, case["target_mask"]))

        torch.manual_seed(200 + i)
        images = model.generate(
            pil_image=ref_image,
            mask_image_0=ref_mask,
            image=target_image,
            mask_image=target_mask,
            prompt="a photo of a pipe interior with a crack",
            num_samples=1,
            num_inference_steps=20,
            strength=0.6,
            scale=1.0,
            guidance_scale=7.5,
            seed=200 + i,
        )

        gen_time = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        if images:
            out_path = os.path.join(OUT_DIR, f"{case['name']}_output.png")
            images[0].save(out_path)
            status = "OK"
        else:
            status = "NO IMAGE RETURNED"

        print(f"  status={status} time={gen_time:.2f}s peak_alloc={peak_alloc:.3f}GB peak_reserved={peak_reserved:.3f}GB")
        results.append({"name": case["name"], "status": status, "time_s": round(gen_time, 2),
                         "peak_allocated_gb": round(peak_alloc, 3), "peak_reserved_gb": round(peak_reserved, 3)})

    print("\n" + "=" * 60)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
