"""
Kraken Discord Command Parser
Parses Discord messages into structured command data
"""

import re
import shlex
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class ParsedCommand:
    """Structured representation of a parsed Discord command."""

    command: str  # The command name (e.g., "generate", "help")
    prompt: str  # The main prompt text
    raw_message: str  # Original message content

    # Optional parameters with flags
    negative: str = ""
    steps: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    style: Optional[str] = None
    model: Optional[str] = None

    # Image attachment info
    has_image: bool = False
    image_url: Optional[str] = None

    # Metadata
    user_id: int = 0
    user_name: str = ""
    channel_id: int = 0
    message_id: int = 0

    # Prompt modifiers (quick tags)
    modifiers: List[str] = field(default_factory=list)

    # Enhancement flag
    enhance: bool = False

    # Upscale flag
    upscale: bool = False

    # Raw parsed arguments for extensibility
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with fallback to default."""
        value = getattr(self, key, None)
        if value is not None:
            return value
        return self.extra_args.get(key, default)


class CommandParser:
    """
    Parses Discord command messages into structured data.

    Supports format: !command prompt text --flag value --another value
    """

    # Recognized flags and their types
    FLAG_TYPES = {
        "negative": str,
        "neg": str,  # alias for negative
        "n": str,    # short alias
        "steps": int,
        "s": int,    # short alias
        "cfg": float,
        "c": float,  # short alias
        "width": int,
        "w": int,    # short alias
        "height": int,
        "h": int,    # short alias
        "seed": int,
        "style": str,
        "model": str,
        "m": str,    # short alias for model
    }

    # Map short aliases to full names
    ALIASES = {
        "neg": "negative",
        "n": "negative",
        "s": "steps",
        "c": "cfg",
        "w": "width",
        "h": "height",
        "m": "model",
    }

    # Common size presets for easy access
    SIZE_PRESETS = {
        "square": (1024, 1024),
        "landscape": (1216, 832),
        "portrait": (832, 1216),
        "wide": (1344, 768),
        "tall": (768, 1344),
        "phone": (832, 1216),
        "desktop": (1216, 832),
    }

    # Quick modifier flags (boolean flags that add prompt tags)
    # Format: flag_name -> (prefix_tag, suffix_tag)
    MODIFIER_TAGS = {
        # Quality modifiers
        "masterpiece": ("masterpiece, best quality,", ""),
        "detailed": ("", ", highly detailed, intricate details"),
        "hd": ("", ", 8k uhd, high resolution, sharp focus"),
        "professional": ("professional", ", high quality, expert craftsmanship"),

        # Lighting modifiers
        "cinematic": ("cinematic film still,", ", cinematic lighting, dramatic shadows, film grain, anamorphic"),
        "dramatic": ("", ", dramatic lighting, dynamic shadows, high contrast"),
        "soft": ("", ", soft lighting, gentle shadows, diffused light"),
        "golden": ("", ", golden hour lighting, warm tones, sun flare"),
        "neon": ("", ", neon lighting, cyberpunk glow, vibrant colors"),
        "backlit": ("", ", backlighting, rim light, silhouette"),

        # Atmosphere modifiers
        "fog": ("", ", atmospheric fog, misty, hazy atmosphere"),
        "mist": ("", ", light mist, ethereal atmosphere"),
        "rain": ("", ", rainy atmosphere, wet surfaces, rain drops"),
        "snow": ("", ", snowing, winter atmosphere, frost"),

        # Time of day
        "night": ("", ", nighttime, dark atmosphere, moonlight"),
        "day": ("", ", daytime, bright daylight, clear sky"),
        "sunset": ("", ", sunset, orange sky, warm colors, dusk"),
        "dawn": ("", ", dawn, early morning, soft pink light"),

        # Style modifiers
        "epic": ("epic", ", grand scale, awe-inspiring, breathtaking"),
        "dark": ("", ", dark and moody, ominous atmosphere, shadows"),
        "vibrant": ("", ", vibrant colors, colorful, saturated"),
        "muted": ("", ", muted colors, desaturated, subtle tones"),
        "vintage": ("", ", vintage aesthetic, retro, nostalgic, film grain"),
        "noir": ("", ", film noir style, black and white, high contrast, shadows"),

        # Camera/technical
        "bokeh": ("", ", bokeh, shallow depth of field, blurred background"),
        "dof": ("", ", depth of field, focus blur, cinematic focus"),
        "wideangle": ("", ", wide angle lens, expansive view"),
        "macro": ("", ", macro photography, extreme close-up, detailed"),
        "portrait": ("portrait shot of", ", portrait photography, shallow dof"),

        # Artistic
        "painterly": ("", ", painterly style, artistic brushstrokes"),
        "ethereal": ("", ", ethereal, dreamlike, otherworldly"),
        "gritty": ("", ", gritty, raw, textured, realistic"),
        "clean": ("", ", clean lines, polished, refined"),
    }

    def __init__(self, prefix: str = "!"):
        """
        Initialize parser with command prefix.

        Args:
            prefix: Command prefix (default "!")
        """
        self.prefix = prefix

    def parse(self, message_content: str, attachments: Optional[List[Any]] = None) -> Optional[ParsedCommand]:
        """
        Parse a Discord message into a structured command.

        Args:
            message_content: The raw message text
            attachments: List of Discord attachment objects (optional)

        Returns:
            ParsedCommand if valid command, None otherwise
        """
        content = message_content.strip()

        # Must start with prefix
        if not content.startswith(self.prefix):
            return None

        # Remove prefix
        content = content[len(self.prefix):]

        # Extract command name (first word)
        parts = content.split(maxsplit=1)
        if not parts:
            return None

        command = parts[0].lower()
        remainder = parts[1] if len(parts) > 1 else ""

        # Parse the remainder for prompt and flags
        prompt, flags, modifiers, enhance, upscale = self._parse_flags(remainder)

        # Process flag aliases
        resolved_flags = {}
        for key, value in flags.items():
            resolved_key = self.ALIASES.get(key, key)
            resolved_flags[resolved_key] = value

        # Handle size preset if width/height is a preset name
        width = resolved_flags.get("width")
        height = resolved_flags.get("height")

        if isinstance(width, str) and width in self.SIZE_PRESETS:
            width, height = self.SIZE_PRESETS[width]
            resolved_flags["width"] = width
            resolved_flags["height"] = height

        # Check for image attachments
        has_image = False
        image_url = None
        if attachments:
            for att in attachments:
                # Check if it's an image
                content_type = getattr(att, 'content_type', '') or ''
                filename = getattr(att, 'filename', '') or ''
                if content_type.startswith('image/') or any(
                    filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']
                ):
                    has_image = True
                    image_url = getattr(att, 'url', None)
                    break

        # Build ParsedCommand
        result = ParsedCommand(
            command=command,
            prompt=prompt.strip(),
            raw_message=message_content,
            negative=resolved_flags.get("negative", ""),
            steps=self._safe_int(resolved_flags.get("steps")),
            cfg=self._safe_float(resolved_flags.get("cfg")),
            width=self._safe_int(resolved_flags.get("width")),
            height=self._safe_int(resolved_flags.get("height")),
            seed=self._safe_int(resolved_flags.get("seed")),
            style=resolved_flags.get("style"),
            model=resolved_flags.get("model"),
            has_image=has_image,
            image_url=image_url,
            modifiers=modifiers,
            enhance=enhance,
            upscale=upscale,
            extra_args={k: v for k, v in resolved_flags.items()
                       if k not in ["negative", "steps", "cfg", "width", "height", "seed", "style", "model"]},
        )

        return result

    def _parse_flags(self, text: str) -> Tuple[str, Dict[str, Any], List[str], bool, bool]:
        """
        Parse text into prompt, flag dictionary, and modifiers.

        Args:
            text: The text after the command name

        Returns:
            Tuple of (prompt_text, flags_dict, modifiers_list, enhance_bool, upscale_bool)
        """
        if not text:
            return "", {}, [], False, False

        flags = {}
        modifiers = []
        enhance = False
        upscale = False

        # Split on -- flags while preserving quoted strings
        # Pattern matches --flag value or --flag "quoted value" or --flag (boolean)
        # IMPORTANT: The value part must NOT match things starting with -- (other flags)
        # Using negative lookahead (?!--) to prevent capturing the next flag as a value
        pattern = r'--(\w+)(?:\s+(?:"([^"]+)"|(?!--)(\S+)))?'

        # Find all flag matches and their positions
        matches = list(re.finditer(pattern, text))

        if not matches:
            # No flags, entire text is prompt
            return text, {}, [], False, False

        # Text before first flag is the prompt
        first_match_start = matches[0].start()
        prompt = text[:first_match_start].strip()

        # Extract flag values
        for match in matches:
            flag_name = match.group(1).lower()
            # Value is either quoted (group 2) or unquoted (group 3)
            value = match.group(2) if match.group(2) else match.group(3)

            # Check if it's an enhance flag
            if flag_name == "enhance":
                enhance = True
                continue

            # Check if it's an upscale flag
            if flag_name == "upscale":
                upscale = True
                continue

            # Check if it's a size preset shorthand flag
            if flag_name in self.SIZE_PRESETS:
                w, h = self.SIZE_PRESETS[flag_name]
                flags["width"] = w
                flags["height"] = h
                continue

            # Check if it's a modifier flag (boolean tag)
            if flag_name in self.MODIFIER_TAGS:
                modifiers.append(flag_name)
                continue

            if value is None:
                # Flag without value, treat as boolean
                value = True

            # Convert to appropriate type
            expected_type = self.FLAG_TYPES.get(flag_name)
            if expected_type and value is not True:
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    pass  # Keep as string if conversion fails

            flags[flag_name] = value

        return prompt, flags, modifiers, enhance, upscale

    def apply_modifiers(self, prompt: str, modifiers: List[str]) -> str:
        """
        Apply modifier tags to a prompt.

        Args:
            prompt: The base prompt
            modifiers: List of modifier names to apply

        Returns:
            Modified prompt with tags added
        """
        if not modifiers:
            return prompt

        prefixes = []
        suffixes = []

        for mod in modifiers:
            if mod in self.MODIFIER_TAGS:
                prefix, suffix = self.MODIFIER_TAGS[mod]
                if prefix:
                    prefixes.append(prefix)
                if suffix:
                    suffixes.append(suffix)

        # Build final prompt
        result = prompt
        if prefixes:
            result = " ".join(prefixes) + " " + result
        if suffixes:
            result = result + " ".join(suffixes)

        return result

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def get_help_text(self) -> str:
        """Generate help text for available commands."""
        return """**Kraken Discord Bot - Commands**

**!generate <prompt>** - Generate an image from text
  Example: `!generate a majestic dragon flying over mountains`

**!img2img <prompt>** - Transform an attached image (attach an image!)
  Example: `!img2img make it look like a painting` (with image attached)

**!help** - Show this help message

**Parameter Flags:**
  `--negative <text>` - What to avoid (e.g., `--negative blurry, ugly`)
  `--steps <1-50>` - Generation steps (default: 20)
  `--cfg <1-20>` - Guidance scale (default: 7)
  `--seed <number>` - Specific seed for reproducibility
  `--style <name>` - Style preset (see below)
  `--width <pixels>` - Image width (default: 1024)
  `--height <pixels>` - Image height (default: 1024)
  `--enhance` - Use AI to expand your prompt (slower but better results)
  `--upscale` - Upscale the image (if upscale model is configured)

**Quick Modifier Flags** (add quality/style tags instantly):
  *Quality:* `--masterpiece` `--detailed` `--hd` `--professional`
  *Lighting:* `--cinematic` `--dramatic` `--soft` `--golden` `--neon` `--backlit`
  *Atmosphere:* `--fog` `--mist` `--rain` `--snow`
  *Time:* `--night` `--day` `--sunset` `--dawn`
  *Style:* `--epic` `--dark` `--vibrant` `--muted` `--vintage` `--noir`
  *Camera:* `--bokeh` `--dof` `--wideangle` `--macro` `--portrait`
  *Artistic:* `--painterly` `--ethereal` `--gritty` `--clean`

**Size Presets:** (use directly as flags!)
  `--square` (1024x1024), `--landscape` (1216x832), `--portrait` (832x1216)
  `--wide` (1344x768), `--tall` (768x1344), `--phone`, `--desktop`

**Style Presets:**
  `photorealistic` `anime` `fantasy` `scifi` `artistic` `cinematic` `cute` `dark` `vintage` `minimalist`

**Examples:**
  `!generate dracula's castle --cinematic --fog --night --masterpiece`
  `!generate a cyberpunk city --style scifi --neon --rain --detailed`
  `!generate cute cat --soft --vibrant --bokeh`
  `!generate epic battle scene --enhance` (uses AI to expand prompt)
  `!generate landscape photo --upscale` (generates and upscales)
"""
