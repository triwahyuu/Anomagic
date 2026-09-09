#!/usr/bin/env bash
set -e

git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" >> ~/.bashrc
echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.zsh_history" >> ~/.zshrc
echo "setopt histignorealldups" >> ~/.zshrc
sudo chown -R "$USER" /commandhistory

echo "export PATH=/home/ubuntu/.local/bin:$PATH" >> ~/.bashrc
echo "export PATH=/home/ubuntu/.local/bin:$PATH" >> ~/.zshrc

# fix claude permission
sudo chown -R "$USER" "$HOME/.claude"

echo ""
echo "=== Anomagic environment sanity check ==="
python -c "
import torch, diffusers, transformers, peft, cv2
print('torch          ', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('torchvision    ', __import__('torchvision').__version__)
print('diffusers      ', diffusers.__version__)
print('transformers   ', transformers.__version__)
print('peft           ', peft.__version__)
print('opencv         ', cv2.__version__)
import huggingface_hub
print('huggingface_hub', huggingface_hub.__version__, '(must stay 0.23.2 -- diffusers==0.22.1 breaks on newer)')
"
echo "=================================================="
echo ""
echo "Software dependencies are fully installed (/opt/venv). NOT baked into the image (large,"
echo "~6.44GB total for the minimal verified inference set -- see playground/handover for the"
echo "full breakdown and the two patches this repo needs before it runs):"
echo ""
echo "  huggingface-cli download yuxinjiang11/Anomagic_model checkpoint/anomagic.bin checkpoint/attention_module.bin --local-dir ."
echo "  huggingface-cli download h94/IP-Adapter models/image_encoder/model.safetensors models/image_encoder/config.json --local-dir ip_adapter_hf"
echo "  huggingface-cli download stabilityai/sd-vae-ft-mse diffusion_pytorch_model.safetensors config.json --local-dir vae_hf"
echo "  huggingface-cli download SG161222/Realistic_Vision_V4.0_noVAE text_encoder/model.safetensors text_encoder/config.json tokenizer/* unet/config.json scheduler/scheduler_config.json --local-dir base_hf"
echo ""
echo "Then run: python run_infer_test.py  (real end-to-end inference + VRAM measurement)"
echo "      or: python run_train_step_test.py --precision fp16 --batch_size 2  (training-step VRAM measurement)"
echo ""
echo "DONE!"
