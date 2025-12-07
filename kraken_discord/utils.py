"""
Kraken Discord Utilities
Helper functions for image conversion, embeds, etc.
"""

import io
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI image tensor to PIL Image.

    Args:
        tensor: Image tensor of shape (H, W, C) or (B, H, W, C)

    Returns:
        PIL Image
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Convert to numpy
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    # Convert from float [0,1] to uint8 [0,255]
    array = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(array)


def image_to_bytes(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    format: str = "PNG",
    quality: int = 95
) -> bytes:
    """
    Convert image to bytes for Discord upload.

    Args:
        image: Image as tensor, PIL Image, or numpy array
        format: Output format (PNG, WEBP, JPEG)
        quality: Quality for lossy formats (1-100)

    Returns:
        Image bytes
    """
    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        pil_image = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Ensure RGB mode for JPEG
    if format.upper() == "JPEG" and pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")

    # Save to bytes buffer
    buffer = io.BytesIO()
    save_kwargs = {}

    if format.upper() in ("WEBP", "JPEG"):
        save_kwargs["quality"] = quality
    if format.upper() == "PNG":
        save_kwargs["compress_level"] = 6

    pil_image.save(buffer, format=format, **save_kwargs)
    buffer.seek(0)

    return buffer.getvalue()


def create_embed(
    title: str,
    description: str = "",
    color: int = 0x9B59B6,  # Purple - Kraken theme
    fields: Optional[List[Dict[str, Any]]] = None,
    footer: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    image_url: Optional[str] = None,
    timestamp: bool = False
) -> Dict[str, Any]:
    """
    Create a Discord embed dictionary.

    Args:
        title: Embed title
        description: Embed description
        color: Embed color (hex integer)
        fields: List of field dicts with name, value, inline
        footer: Footer text
        thumbnail_url: Thumbnail image URL
        image_url: Main image URL
        timestamp: Whether to include current timestamp

    Returns:
        Embed dictionary for Discord API
    """
    embed = {
        "title": title,
        "description": description,
        "color": color,
    }

    if fields:
        embed["fields"] = [
            {
                "name": f.get("name", ""),
                "value": str(f.get("value", "")),
                "inline": f.get("inline", False)
            }
            for f in fields
        ]

    if footer:
        embed["footer"] = {"text": footer}

    if thumbnail_url:
        embed["thumbnail"] = {"url": thumbnail_url}

    if image_url:
        embed["image"] = {"url": image_url}

    if timestamp:
        embed["timestamp"] = datetime.utcnow().isoformat()

    return embed


def create_generation_embed(
    prompt: str,
    negative: str = "",
    seed: int = 0,
    steps: int = 20,
    cfg: float = 7.0,
    size: str = "1024x1024",
    style: str = "none",
    model: str = "",
    generation_time: Optional[float] = None,
    user_name: str = ""
) -> Dict[str, Any]:
    """
    Create an embed with generation parameters.

    Args:
        prompt: The generation prompt
        negative: Negative prompt
        seed: Generation seed
        steps: Number of steps
        cfg: CFG scale
        size: Image size string
        style: Style preset name
        model: Model name
        generation_time: Time taken in seconds
        user_name: Requesting user's name

    Returns:
        Embed dictionary
    """
    fields = []

    # Truncate long prompts for embed
    display_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt

    if negative:
        display_neg = negative[:100] + "..." if len(negative) > 100 else negative
        fields.append({"name": "Negative", "value": display_neg, "inline": False})

    fields.extend([
        {"name": "Seed", "value": str(seed), "inline": True},
        {"name": "Steps", "value": str(steps), "inline": True},
        {"name": "CFG", "value": str(cfg), "inline": True},
        {"name": "Size", "value": size, "inline": True},
    ])

    if style and style != "none":
        fields.append({"name": "Style", "value": style.title(), "inline": True})

    if generation_time:
        fields.append({"name": "Time", "value": f"{generation_time:.1f}s", "inline": True})

    footer = f"Requested by {user_name}" if user_name else "Kraken Discord"

    if model:
        # Extract just model name from path
        model_name = model.split("/")[-1].split("\\")[-1]
        model_name = model_name.replace(".safetensors", "").replace(".ckpt", "")
        footer += f" | Model: {model_name}"

    return create_embed(
        title="Generated Image",
        description=f"**Prompt:** {display_prompt}",
        fields=fields,
        footer=footer,
        timestamp=True
    )


def format_error_message(error: str) -> str:
    """
    Format an error message for Discord display.

    Args:
        error: Error message

    Returns:
        Formatted error string
    """
    return f"**Error:** {error}\n\nUse `!help` for usage information."


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to max length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def validate_dimensions(width: int, height: int, max_pixels: int = 2048 * 2048) -> tuple:
    """
    Validate and potentially adjust image dimensions.

    Args:
        width: Requested width
        height: Requested height
        max_pixels: Maximum total pixels allowed

    Returns:
        Tuple of (validated_width, validated_height, was_adjusted)
    """
    # Clamp to reasonable ranges
    width = max(64, min(2048, width))
    height = max(64, min(2048, height))

    # Check total pixels
    if width * height > max_pixels:
        # Scale down proportionally
        scale = (max_pixels / (width * height)) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
        return width, height, True

    # Round to 8 (for latent compatibility)
    width = (width // 8) * 8
    height = (height // 8) * 8

    return width, height, False


def download_image(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """
    Download an image from URL.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        PIL Image or None if failed
    """
    try:
        import requests
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"[KrakenDiscord] Failed to download image: {e}")
        return None


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI tensor format.

    Args:
        image: PIL Image

    Returns:
        Tensor of shape (1, H, W, 3)
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy then tensor
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)

    # Add batch dimension
    return tensor.unsqueeze(0)
