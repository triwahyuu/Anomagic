"""
Real forward+backward+optimizer-step VRAM measurement for Anomagic training, reproducing
research/models/anomagic.md SS9.4's methodology in this fresh sample/anomagic clone.

Reuses Anomagic_train.py's own MyDataset / collate_fn / SelfAttention / Anomagic(nn.Module) /
load_lora_model / encode_long_text verbatim (imported as a module, not retyped, to avoid drift).
The one deliberate deviation from a literal `python Anomagic_train.py` run: the UNet is built
from the base repo's config only (no 3.2GB base-UNet weight download -- architecture/dtype/
activation shapes drive memory, not weight values, so this is valid for a memory measurement,
though NOT a claim about training quality/convergence -- same reasoning anomagic.md SS9.4 used).

Usage:
    .venv/bin/python run_train_step_test.py --precision fp32 --batch_size 1
    .venv/bin/python run_train_step_test.py --precision fp16 --batch_size 4
"""
import argparse
import itertools
import os
import sys

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Anomagic_train as T  # noqa: E402  (real repo module, reused verbatim)


def build_unet_from_config(base_hf_dir, device, dtype):
    cfg = UNet2DConditionModel.load_config(os.path.join(base_hf_dir, "unet"))
    unet = UNet2DConditionModel.from_config(cfg).to(device, dtype=dtype)
    return unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    device = "cuda"
    weight_dtype = torch.float16 if args.precision == "fp16" else torch.float32

    base_hf = "base_hf"
    image_encoder_path = "ip_adapter_hf/models/image_encoder"
    ip_adapter_init_ckpt = "ip_adapter_hf/models/ip-adapter_sd15.bin"

    print(f"[{args.precision} bs={args.batch_size}] Building components...")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_hf, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(base_hf, "text_encoder")).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained("vae_hf").to(device, dtype=weight_dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device, dtype=weight_dtype)
    unet = build_unet_from_config(base_hf, device, weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(base_hf, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    image_proj_model = T.ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    from ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)

    unet, lora_layers = T.load_lora_model(unet, device, 4e-4)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    anomagic_model = T.Anomagic(unet, image_proj_model, adapter_modules, ip_adapter_init_ckpt)
    attention_module = T.SelfAttention(1280)

    unet.to(device, dtype=weight_dtype)
    anomagic_model.image_proj_model.to(device, dtype=weight_dtype)
    anomagic_model.adapter_modules.to(device, dtype=weight_dtype)
    attention_module.to(device, dtype=weight_dtype)

    params_to_opt = itertools.chain(
        anomagic_model.image_proj_model.parameters(),
        anomagic_model.adapter_modules.parameters(),
        lora_layers,
        attention_module.parameters(),
    )
    optimizer = torch.optim.AdamW(params_to_opt, lr=1e-4, weight_decay=1e-2)

    print("Building real-data dataloader from MVTec capsule crack stand-in...")
    train_dataset = T.MyDataset(
        "train_data_capsule_crack.json",
        "/workspaces/kdw-pipecrack/datasets/mvtec_ad/capsule",
        tokenizer=tokenizer,
        size=512,
        use_analysis_text=False,
        use_short_text=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=T.collate_fn, num_workers=0,
    )

    torch.cuda.reset_peak_memory_stats()
    anomagic_model.train()
    attention_module.train()

    step = 0
    for batch in train_dataloader:
        if step >= args.steps:
            break
        with torch.no_grad():
            latents = vae.encode(batch["images"].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            outputs = image_encoder(batch["clip_images"].to(device, dtype=weight_dtype))
            image_embeds_raw = outputs.image_embeds
            last_feature_layer_output = outputs.last_hidden_state
            image_embeds_ = []
            for image_embed, drop in zip(image_embeds_raw, batch["drop_image_embeds"]):
                image_embeds_.append(torch.zeros_like(image_embed) if drop == 1 else image_embed)
            encoder_hidden_states = T.encode_long_text(
                batch["text_input_ids"].to(device), tokenizer, text_encoder, device=device
            )

        masks = batch["masks"].to(device, dtype=weight_dtype)
        image_embeds = attention_module(last_feature_layer_output[:, :256, :], masks.unsqueeze(1))
        noise_pred = anomagic_model(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
        loss = (torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                * masks.unsqueeze(1)).mean([1, 2, 3]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step} loss={loss.item():.4f} "
              f"peak_allocated={torch.cuda.max_memory_allocated()/1e9:.3f}GB "
              f"peak_reserved={torch.cuda.max_memory_reserved()/1e9:.3f}GB")
        step += 1

    print("=" * 60)
    print(f"[{args.precision} bs={args.batch_size}] FINAL peak_allocated="
          f"{torch.cuda.max_memory_allocated()/1e9:.3f}GB peak_reserved={torch.cuda.max_memory_reserved()/1e9:.3f}GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
