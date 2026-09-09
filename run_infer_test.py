"""
Real, end-to-end Anomagic inference test using the low-level Anomagic.generate() API.

Bypasses the documented Anomagic_test.py workflow entirely (no Doubao similarity-retrieval
call, no MetaUAS quality gate, no MVTec/VisA dataset-folder plumbing) -- this is the
lightest-weight path identified in research/models/anomagic.md SS4/SS9.3.

Stand-in data (no real KDW pipe photos exist in this repo yet):
  - target normal image : MVTec capsule train/good/000.png
  - reference anomaly    : MVTec capsule test/crack/000.png + ground_truth/crack/000_mask.png
  - target inpaint mask  : generated via mask/creatMask.py::generate_adaptive_mask(), forced to
                           the new 'crack' shape primitive (random-walk / midpoint-displacement,
                           see mask/creatMask.py's generate_crack_mask()) via shape_pool=['crack'].

Because the reference anomaly and the target image are both "capsule", this run does NOT
exercise cross-category transfer -- it validates the generation mechanism + patches only.
"""
import argparse
import os
import sys
import time
import random

import json

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask.creatMask import generate_adaptive_mask

MVTEC_CAPSULE = "/workspaces/kdw-pipecrack/datasets/mvtec_ad/capsule"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"
SIZE = 512


def load_rgb(path, size=SIZE):
    return Image.open(path).convert("RGB").resize((size, size))


def load_mask(path, size=SIZE):
    m = Image.open(path).convert("L").resize((size, size))
    arr = (np.array(m) > 128).astype(np.uint8) * 255
    return Image.fromarray(arr)


def shipped_procedural_mask(target_image_path, category="capsule", seed=0, size=SIZE, shape_pool=None):
    """Uses mask/creatMask.py::generate_adaptive_mask() -- gated on an Otsu-thresholded
    foreground template of the target image, exactly as process_images() builds it (mask/
    creatMask.py lines ~195-207), using the real per-category width/height range from
    mask/mvtec.json. `shape_pool=['crack']` forces the new crack-shaped primitive (see
    mask/creatMask.py::generate_crack_mask()); pass None for the default 5-way random mix."""
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

    with open(os.path.join(os.path.dirname(__file__), "mask", "mvtec.json")) as f:
        mask_info = json.load(f)
    info = mask_info[category]["Anomaly"]

    mask_256 = generate_adaptive_mask(template, info, shape_pool=shape_pool)
    mask_img = Image.fromarray(mask_256).resize((size, size), Image.NEAREST)
    return mask_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-shape", choices=["mixed", "crack"], default="mixed",
                     help="'crack' forces mask/creatMask.py's new crack primitive; "
                          "'mixed' uses the default 5-way random shape choice.")
    ap.add_argument("--out-suffix", default="", help="appended to output filenames")
    args = ap.parse_args()
    shape_pool = ["crack"] if args.mask_shape == "crack" else None
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU idle check: {torch.cuda.memory_allocated() / 1e9:.3f} GB allocated before start")

    from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from ip_adapter.ip_adapter_anomagic import Anomagic

    base_hf = "base_hf"
    vae_hf = "vae_hf"
    image_encoder_path = "ip_adapter_hf/models/image_encoder"

    print("Building tokenizer/text_encoder (real weights)...")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_hf, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(base_hf, "text_encoder")).to(DEVICE, dtype=torch.float16)

    print("Building VAE (real weights)...")
    vae = AutoencoderKL.from_pretrained(vae_hf).to(DEVICE, dtype=torch.float16)

    print("Building UNet from config only (no weight download -- anomagic.bin overwrites it via strict=True load)...")
    unet_config = UNet2DConditionModel.load_config(os.path.join(base_hf, "unet"))
    unet = UNet2DConditionModel.from_config(unet_config).to(DEVICE, dtype=torch.float16)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = StableDiffusionInpaintPipelineLegacy(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"After base pipeline build: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB allocated (peak)")

    print("Constructing Anomagic (real CLIP-ViT-H/14 + anomagic.bin + attention_module.bin)...")
    model = Anomagic(
        pipe,
        image_encoder_path,
        "checkpoint/anomagic.bin",
        "checkpoint/attention_module.bin",
        DEVICE,
    )
    print(f"After Anomagic() construction: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB allocated (peak)")

    print("Loading real MVTec capsule stand-in data...")
    target_image_path = os.path.join(MVTEC_CAPSULE, "train/good/000.png")
    target_image = load_rgb(target_image_path)
    ref_image = load_rgb(os.path.join(MVTEC_CAPSULE, "test/crack/000.png"))
    ref_mask = load_mask(os.path.join(MVTEC_CAPSULE, "ground_truth/crack/000_mask.png"))
    target_mask = shipped_procedural_mask(target_image_path, category="capsule", seed=42, shape_pool=shape_pool)

    target_image.save(os.path.join(OUT_DIR, "input_target_normal.png"))
    ref_image.save(os.path.join(OUT_DIR, "input_reference_anomaly.png"))
    ref_mask.save(os.path.join(OUT_DIR, "input_reference_mask.png"))
    target_mask.save(os.path.join(OUT_DIR, f"input_target_mask{suffix}.png"))

    print("Running Anomagic.generate() ...")
    gen_start = time.time()
    torch.manual_seed(42)
    images = model.generate(
        pil_image=ref_image,
        mask_image_0=ref_mask,
        image=target_image,
        mask_image=target_mask,
        prompt="a photo of a defective object, crack",
        num_samples=1,
        num_inference_steps=20,
        strength=0.6,
        scale=1.0,
        guidance_scale=7.5,
        seed=42,
    )
    gen_time = time.time() - gen_start

    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    if images:
        out_path = os.path.join(OUT_DIR, f"output_sample{suffix}.png")
        images[0].save(out_path)
        print(f"Saved output to {out_path}")
    else:
        print("!! generate() returned no images")

    print("=" * 60)
    print(f"generate() wall time : {gen_time:.2f}s")
    print(f"Peak allocated        : {peak_allocated:.3f} GB")
    print(f"Peak reserved         : {peak_reserved:.3f} GB")
    print(f"Total wall time       : {time.time() - t0:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
