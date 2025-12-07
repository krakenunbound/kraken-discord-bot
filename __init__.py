# -*- coding: utf-8 -*-
"""
Kraken Discord Bot - ComfyUI Custom Node
A standalone Discord bot for AI image generation via ComfyUI.

GitHub: https://github.com/yourusername/kraken-discord-bot
"""

import torch

# --- Performance Backend Hints (Windows & Torch 2.x friendly) ---
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(True)
    sdp_kernel.enable_mem_efficient_sdp(True)
    sdp_kernel.enable_math_sdp(True)
except Exception:
    pass

# Import node classes
from .nodes import KrakenDiscordBot, KrakenDiscordBotStatus

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KrakenDiscordBot": KrakenDiscordBot,
    "KrakenDiscordBotStatus": KrakenDiscordBotStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KrakenDiscordBot": "Kraken Discord Bot (All-in-One)",
    "KrakenDiscordBotStatus": "Kraken Discord Bot Status",
}

# Web directory for custom styling
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
