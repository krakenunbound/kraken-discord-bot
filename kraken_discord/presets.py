"""
Kraken Discord Style Presets
Beginner-friendly style presets that modify prompts
"""

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class StylePreset:
    """A style preset with prompt modifications."""

    name: str
    display_name: str
    description: str
    prompt_prefix: str  # Added before user prompt
    prompt_suffix: str  # Added after user prompt
    negative_prompt: str  # Added to negative prompt
    # Optional parameter overrides
    cfg_override: Optional[float] = None
    steps_override: Optional[int] = None


class StylePresets:
    """
    Manager for beginner-friendly style presets.
    These use prompt engineering rather than LoRAs for simplicity.
    """

    PRESETS: Dict[str, StylePreset] = {
        "none": StylePreset(
            name="none",
            display_name="None",
            description="No style modification",
            prompt_prefix="",
            prompt_suffix="",
            negative_prompt="",
        ),
        "photorealistic": StylePreset(
            name="photorealistic",
            display_name="Photo Realistic",
            description="Realistic photograph style",
            prompt_prefix="photorealistic, highly detailed photograph of",
            prompt_suffix=", professional photography, 8k uhd, sharp focus, natural lighting",
            negative_prompt="cartoon, anime, illustration, painting, drawing, artificial, fake, unrealistic, deformed",
            cfg_override=7.5,
        ),
        "anime": StylePreset(
            name="anime",
            display_name="Anime",
            description="Japanese anime/manga style",
            prompt_prefix="anime style illustration of",
            prompt_suffix=", vibrant colors, clean lines, anime aesthetic, detailed",
            negative_prompt="photorealistic, photograph, 3d render, realistic, western cartoon",
            cfg_override=7.0,
        ),
        "fantasy": StylePreset(
            name="fantasy",
            display_name="Fantasy",
            description="Epic fantasy art style",
            prompt_prefix="epic fantasy art of",
            prompt_suffix=", magical atmosphere, dramatic lighting, detailed fantasy illustration, artstation quality",
            negative_prompt="modern, contemporary, photograph, mundane, boring, simple",
            cfg_override=8.0,
        ),
        "scifi": StylePreset(
            name="scifi",
            display_name="Sci-Fi",
            description="Science fiction style",
            prompt_prefix="science fiction concept art of",
            prompt_suffix=", futuristic, high tech, cinematic lighting, detailed sci-fi illustration",
            negative_prompt="medieval, fantasy, ancient, old fashioned, low tech, primitive",
            cfg_override=7.5,
        ),
        "artistic": StylePreset(
            name="artistic",
            display_name="Artistic",
            description="Painterly artistic style",
            prompt_prefix="beautiful artistic painting of",
            prompt_suffix=", masterful brushwork, artistic interpretation, fine art, expressive",
            negative_prompt="photograph, photorealistic, 3d render, digital art, simple, plain",
            cfg_override=8.0,
        ),
        "cinematic": StylePreset(
            name="cinematic",
            display_name="Cinematic",
            description="Movie-like cinematic style",
            prompt_prefix="cinematic shot of",
            prompt_suffix=", movie still, dramatic lighting, film grain, anamorphic, depth of field, cinematic color grading",
            negative_prompt="flat, boring, amateur, snapshot, overexposed, underexposed",
            cfg_override=7.0,
        ),
        "cute": StylePreset(
            name="cute",
            display_name="Cute",
            description="Adorable cute style",
            prompt_prefix="adorable cute",
            prompt_suffix=", kawaii, charming, lovable, soft lighting, pastel colors, heartwarming",
            negative_prompt="scary, horror, dark, creepy, ugly, gross, disturbing",
            cfg_override=6.5,
        ),
        "dark": StylePreset(
            name="dark",
            display_name="Dark",
            description="Dark and moody style",
            prompt_prefix="dark atmospheric",
            prompt_suffix=", moody lighting, shadows, mysterious, dramatic contrast, noir aesthetic",
            negative_prompt="bright, cheerful, colorful, happy, sunny, light",
            cfg_override=8.0,
        ),
        "vintage": StylePreset(
            name="vintage",
            display_name="Vintage",
            description="Retro vintage style",
            prompt_prefix="vintage retro style",
            prompt_suffix=", nostalgic, aged aesthetic, classic, timeless, faded colors",
            negative_prompt="modern, futuristic, contemporary, digital, new",
            cfg_override=7.0,
        ),
        "minimalist": StylePreset(
            name="minimalist",
            display_name="Minimalist",
            description="Clean minimalist style",
            prompt_prefix="minimalist",
            prompt_suffix=", clean design, simple, elegant, uncluttered, modern aesthetic",
            negative_prompt="busy, cluttered, complex, detailed, ornate, chaotic",
            cfg_override=6.0,
        ),
    }

    @classmethod
    def get_preset(cls, name: str) -> StylePreset:
        """
        Get a style preset by name.

        Args:
            name: Preset name (case insensitive)

        Returns:
            StylePreset, defaults to 'none' if not found
        """
        return cls.PRESETS.get(name.lower(), cls.PRESETS["none"])

    @classmethod
    def get_preset_names(cls) -> List[str]:
        """Get list of all preset names."""
        return list(cls.PRESETS.keys())

    @classmethod
    def get_display_names(cls) -> Dict[str, str]:
        """Get mapping of preset names to display names."""
        return {name: preset.display_name for name, preset in cls.PRESETS.items()}

    @classmethod
    def apply_style(
        cls,
        prompt: str,
        negative: str,
        style_name: str,
        cfg: float,
        steps: int
    ) -> tuple:
        """
        Apply a style preset to prompt and parameters.

        Args:
            prompt: User's prompt
            negative: User's negative prompt
            style_name: Style preset name
            cfg: Original CFG value
            steps: Original steps value

        Returns:
            Tuple of (modified_prompt, modified_negative, cfg, steps)
        """
        preset = cls.get_preset(style_name)

        # Build modified prompt
        parts = []
        if preset.prompt_prefix:
            parts.append(preset.prompt_prefix)
        parts.append(prompt)
        if preset.prompt_suffix:
            parts.append(preset.prompt_suffix)
        modified_prompt = " ".join(parts)

        # Build modified negative
        neg_parts = []
        if negative:
            neg_parts.append(negative)
        if preset.negative_prompt:
            neg_parts.append(preset.negative_prompt)
        modified_negative = ", ".join(neg_parts) if neg_parts else ""

        # Apply overrides
        final_cfg = preset.cfg_override if preset.cfg_override is not None else cfg
        final_steps = preset.steps_override if preset.steps_override is not None else steps

        return modified_prompt, modified_negative, final_cfg, final_steps

    @classmethod
    def get_styles_help(cls) -> str:
        """Generate help text for styles."""
        lines = ["**Available Styles:**"]
        for name, preset in cls.PRESETS.items():
            if name != "none":
                lines.append(f"  `{name}` - {preset.description}")
        return "\n".join(lines)
