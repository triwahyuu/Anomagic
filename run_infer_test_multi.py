"""
Real inference across 5 additional cases beyond the original capsule-only run
(run_infer_test.py), spanning 4 new MVTec categories + 2 genuine cross-category transfers --
Anomagic's actual headline claim (a reference anomaly from an *unrelated* category
transplanted onto the target), not yet exercised by the capsule-only run.

Data: a handful of specific files downloaded from the Voxel51/mvtec-ad HF mirror (same source
the AnomalyDiffusion session used), NOT the whole dataset -- see extra_mvtec_data/ (organized
as <category>_<defect>/{normal,anomaly,anomaly_mask}.png).

Cases:
  1. hazelnut  -> hazelnut/crack   (same-category baseline, new category)
  2. tile      -> tile/crack       (same-category baseline, new category)
  3. wood      -> wood/scratch     (same-category baseline, new category)
  4. CROSS: capsule target  <- hazelnut/crack reference   (genuine zero-shot transfer)
  5. CROSS: tile target     <- metal_nut/scratch reference (genuine zero-shot transfer)

Target inpaint mask: the new crack-shaped primitive (mask/creatMask.py::generate_crack_mask,
via shape_pool=['crack']) in every case, built from a real Otsu template of each target image,
for methodological consistency with the earlier capsule crack-mask run.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask.creatMask import generate_adaptive_mask

MVTEC_CAPSULE = "/workspaces/kdw-pipecrack/datasets/mvtec_ad/capsule"
EXTRA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_mvtec_data")
OUT_DIR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "multi")

# 'crack' = new primitive only (this session's addition). 'original' = the shipped
# rectangle/ellipse/polygon/irregular-blob primitives only (crack excluded) -- for a direct,
# like-for-like comparison on the exact same target/reference pairs and seeds.
SHAPE_POOLS = {
    "crack": ["crack"],
    "original": ["rectangle", "ellipse", "polygon", "irregular"],
}

DEVICE = "cuda"
SIZE = 512

with open(os.path.join(os.path.dirname(__file__), "mask", "mvtec.json")) as f:
    MASK_INFO = json.load(f)


def load_rgb(path, size=SIZE):
    return Image.open(path).convert("RGB").resize((size, size))


def load_mask(path, size=SIZE):
    m = Image.open(path).convert("L").resize((size, size))
    arr = (np.array(m) > 128).astype(np.uint8) * 255
    return Image.fromarray(arr)


def crack_mask_for(target_image_path, category, seed, shape_pool):
    import random
    random.seed(seed)
    np.random.seed(seed)
    img = cv2.imread(target_image_path)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh[0][0] == 255:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    template = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    info = MASK_INFO[category]["Anomaly"]
    mask_256 = generate_adaptive_mask(template, info, shape_pool=shape_pool)
    return Image.fromarray(mask_256).resize((SIZE, SIZE), Image.NEAREST)


CASES = [
    {
        "name": "1_hazelnut_same",
        "target_image": f"{EXTRA}/hazelnut_crack/normal.png",
        "target_category": "hazelnut",
        "ref_image": f"{EXTRA}/hazelnut_crack/anomaly.png",
        "ref_mask": f"{EXTRA}/hazelnut_crack/anomaly_mask.png",
        "prompt": "a photo of a hazelnut with a crack",
        "cross_category": False,
    },
    {
        "name": "2_tile_same",
        "target_image": f"{EXTRA}/tile_crack/normal.png",
        "target_category": "tile",
        "ref_image": f"{EXTRA}/tile_crack/anomaly.png",
        "ref_mask": f"{EXTRA}/tile_crack/anomaly_mask.png",
        "prompt": "a photo of a tile surface with a crack",
        "cross_category": False,
    },
    {
        "name": "3_wood_same",
        "target_image": f"{EXTRA}/wood_scratch/normal.png",
        "target_category": "wood",
        "ref_image": f"{EXTRA}/wood_scratch/anomaly.png",
        "ref_mask": f"{EXTRA}/wood_scratch/anomaly_mask.png",
        "prompt": "a photo of a wood surface with a scratch",
        "cross_category": False,
    },
    {
        "name": "4_cross_capsule_target_hazelnut_crack_ref",
        "target_image": f"{MVTEC_CAPSULE}/train/good/010.png",
        "target_category": "capsule",
        "ref_image": f"{EXTRA}/hazelnut_crack/anomaly.png",
        "ref_mask": f"{EXTRA}/hazelnut_crack/anomaly_mask.png",
        "prompt": "a photo of a capsule with a crack",
        "cross_category": True,
    },
    {
        "name": "5_cross_tile_target_metalnut_scratch_ref",
        "target_image": f"{EXTRA}/tile_crack/normal.png",
        "target_category": "tile",
        "ref_image": f"{EXTRA}/metal_nut_scratch/anomaly.png",
        "ref_mask": f"{EXTRA}/metal_nut_scratch/anomaly_mask.png",
        "prompt": "a photo of a tile surface with a scratch",
        "cross_category": True,
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-mode", choices=list(SHAPE_POOLS.keys()), default="crack")
    args = ap.parse_args()
    shape_pool = SHAPE_POOLS[args.mask_mode]
    out_dir = os.path.join(OUT_DIR_ROOT, args.mask_mode)
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from ip_adapter.ip_adapter_anomagic import Anomagic

    base_hf = "base_hf"
    vae_hf = "vae_hf"
    image_encoder_path = "ip_adapter_hf/models/image_encoder"

    print("Building shared pipeline components (real weights, loaded once for all 5 cases)...")
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

    results = []
    for i, case in enumerate(CASES):
        print(f"--- Case {case['name']} ---")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        target_image = load_rgb(case["target_image"])
        ref_image = load_rgb(case["ref_image"])
        ref_mask = load_mask(case["ref_mask"])
        target_mask = crack_mask_for(case["target_image"], case["target_category"], seed=100 + i, shape_pool=shape_pool)

        case_dir = os.path.join(out_dir, case["name"])
        os.makedirs(case_dir, exist_ok=True)
        target_image.save(os.path.join(case_dir, "target_normal.png"))
        ref_image.save(os.path.join(case_dir, "reference_anomaly.png"))
        ref_mask.save(os.path.join(case_dir, "reference_mask.png"))
        target_mask.save(os.path.join(case_dir, "target_mask.png"))

        torch.manual_seed(100 + i)
        images = model.generate(
            pil_image=ref_image,
            mask_image_0=ref_mask,
            image=target_image,
            mask_image=target_mask,
            prompt=case["prompt"],
            num_samples=1,
            num_inference_steps=20,
            strength=0.6,
            scale=1.0,
            guidance_scale=7.5,
            seed=100 + i,
        )

        gen_time = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        if images:
            out_path = os.path.join(case_dir, "output.png")
            images[0].save(out_path)
            status = "OK"
        else:
            status = "NO IMAGE RETURNED"

        print(f"  cross_category={case['cross_category']} status={status} "
              f"time={gen_time:.2f}s peak_alloc={peak_alloc:.3f}GB peak_reserved={peak_reserved:.3f}GB")
        results.append({
            "name": case["name"], "cross_category": case["cross_category"], "status": status,
            "time_s": round(gen_time, 2), "peak_allocated_gb": round(peak_alloc, 3),
            "peak_reserved_gb": round(peak_reserved, 3),
        })

    print("\n" + "=" * 70)
    for r in results:
        print(r)
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {os.path.join(out_dir, 'results_summary.json')}")


if __name__ == "__main__":
    main()
