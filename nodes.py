"""
Kraken Discord Bot - All-in-One Generation Node
A single node that handles Discord bot + image generation internally.
Now with LoRA and upscale model support!
"""

import os
import time
import random
import json
import hashlib
import re
import gc
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List

import requests
import folder_paths

# For prompt enhancement with Qwen2-VL-2B
try:
    from transformers import pipeline, AutoProcessor, Qwen2VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[KrakenDiscordBot] transformers not found - prompt enhancement disabled")
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.model_management as model_management
from comfy.samplers import KSampler as ComfyKSampler
from nodes import CLIPTextEncode, LoraLoader

# Try to import model_loading for upscale models
try:
    from spandrel import ModelLoader
    HAS_SPANDREL = True
except ImportError:
    try:
        from comfy_extras.chainner_models import model_loading
        HAS_SPANDREL = False
    except ImportError:
        model_loading = None
        HAS_SPANDREL = False


def get_checkpoint_list():
    """Get list of available checkpoints."""
    try:
        ckpts = folder_paths.get_filename_list("checkpoints")
        return ckpts if ckpts else ["none"]
    except:
        return ["none"]


def get_lora_list():
    """Get list of available LoRAs."""
    try:
        loras = folder_paths.get_filename_list("loras")
        return ["None"] + sorted(loras) if loras else ["None"]
    except:
        return ["None"]


def get_upscale_model_list():
    """Get list of available upscale models."""
    try:
        models = folder_paths.get_filename_list("upscale_models")
        return ["None"] + sorted(models) if models else ["None"]
    except:
        return ["None"]


def get_config_value(key, default):
    """Get a value from config file."""
    try:
        from .kraken_discord.config import get_config
        config = get_config()
        return config.get(key, default)
    except:
        return default


# -------- CivitAI Integration (from kraken_loras3) ---------

CONFIG_PATH = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "user", "kraken_config.json")
CIVITAI_MODEL_VERSION_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash"

def load_civitai_api_key() -> str:
    """Load CivitAI API key from config file."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                return config.get("civitai_api_key", "")
        return ""
    except Exception:
        return ""


def calculate_sha256(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_lora_tags_cache_path():
    """Get path to LoRA tags cache file."""
    try:
        lora_paths = folder_paths.get_folder_paths("loras")
        if lora_paths:
            return os.path.join(lora_paths[0], "loras_tags.json")
    except:
        pass
    return os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "loras_tags.json")


def civitai_fetch_triggers(lora_name: str, api_key: str = None) -> Tuple[List[str], float, float]:
    """
    Fetch trigger words and recommended strengths from CivitAI.
    Returns: (trigger_words, model_strength, clip_strength)
    """
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return [], 1.0, 1.0

    cache_path = get_lora_tags_cache_path()

    # Try to load from cache
    try:
        with open(cache_path, 'r') as f:
            lora_tags = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        lora_tags = {}

    cached_data = lora_tags.get(lora_name, None)
    if cached_data:
        return (
            cached_data.get("trigger_words", []),
            cached_data.get("model_strength", 1.0),
            cached_data.get("clip_strength", 1.0)
        )

    # Fetch from CivitAI
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        lora_hash = calculate_sha256(lora_path)
        r = requests.get(f"{CIVITAI_MODEL_VERSION_BY_HASH}/{lora_hash}", headers=headers, timeout=10)
        r.raise_for_status()
        model_info = r.json()
    except Exception:
        model_info = None

    if model_info:
        trigger_words = model_info.get("trainedWords", [])
        description = model_info.get("description", "") or ""
        model_strength, clip_strength = 1.0, 1.0

        # Try to parse recommended strength from description
        weight_match = re.search(r"(?:weight|strength)\s*[:=]\s*(\d*\.?\d+)(?:-(\d*\.?\d+))?", description, re.IGNORECASE)
        if weight_match:
            model_strength = float(weight_match.group(1))
            clip_strength = float(weight_match.group(2) or model_strength)

        # Cache the results
        lora_tags[lora_name] = {
            "trigger_words": trigger_words,
            "model_strength": model_strength,
            "clip_strength": clip_strength
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(lora_tags, f, indent=4)
        except Exception:
            pass

        return trigger_words, model_strength, clip_strength

    return [], 1.0, 1.0


class KrakenDiscordBot:
    """
    All-in-One Discord Bot for Image Generation

    This single node:
    - Runs a Discord bot
    - Waits for !generate commands
    - Loads the model, applies LoRA (with CivitAI triggers), generates images
    - Upscales with selected model
    - Sends results back to Discord

    Just add this node, set your token, pick a model, and queue!
    """

    # Cached models to avoid reloading
    _cached_model = None
    _cached_model_name = None
    _cached_clip = None
    _cached_vae = None
    _cached_upscale_model = None
    _cached_upscale_model_name = None
    # Cached LLM for prompt enhancement
    _cached_llm_pipeline = None
    _cached_llm_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_checkpoint_list()
        loras = get_lora_list()
        upscale_models = get_upscale_model_list()

        # Get saved values from config
        saved_negative = get_config_value("default_negative", "ugly, blurry, low quality, distorted")
        saved_steps = get_config_value("default_steps", 20)
        saved_cfg = get_config_value("default_cfg", 7.0)
        saved_width = get_config_value("default_width", 1024)
        saved_height = get_config_value("default_height", 1024)

        # Get Discord token - show "Token Loaded" if saved, otherwise empty for input
        saved_token = get_config_value("discord_token", "")
        discord_token_display = "Token Loaded" if saved_token else ""

        # Check CivitAI API key status
        api_key_status = "Key Loaded" if load_civitai_api_key() else "No Key"

        return {
            "required": {
                "discord_token": ("STRING", {
                    "default": discord_token_display,
                    "multiline": False,
                    "placeholder": "Paste Discord bot token here"
                }),
                "checkpoint": (checkpoints, {}),
                "default_negative": ("STRING", {
                    "default": saved_negative,
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "default_steps": ("INT", {
                    "default": saved_steps,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "default_cfg": ("FLOAT", {
                    "default": saved_cfg,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5
                }),
                "default_width": ("INT", {
                    "default": saved_width,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "default_height": ("INT", {
                    "default": saved_height,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "sampler_name": (ComfyKSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (ComfyKSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                # LoRA 1
                "lora_1": (loras, {"default": "None"}),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # LoRA 2
                "lora_2": (loras, {"default": "None"}),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # LoRA 3
                "lora_3": (loras, {"default": "None"}),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # CivitAI
                "use_civitai_triggers": ("BOOLEAN", {"default": True}),
                "civitai_api_status": ("STRING", {"default": api_key_status, "multiline": False}),
                # Upscale Settings
                "upscale_model": (upscale_models, {"default": "None"}),
                # Bot Settings
                "rate_limit_seconds": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 300,
                    "step": 5
                }),
                "max_queue_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "allowed_channels": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Channel IDs (comma-separated, empty=all)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "run_bot"
    CATEGORY = "Kraken/Discord"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute to wait for next command
        return time.time()

    def _load_checkpoint(self, checkpoint_name: str) -> Tuple[Any, Any, Any]:
        """Load checkpoint, using cache if available."""
        if (self._cached_model_name == checkpoint_name and
            self._cached_model is not None):
            print(f"[KrakenDiscordBot] Using cached model: {checkpoint_name}")
            return self._cached_model, self._cached_clip, self._cached_vae

        print(f"[KrakenDiscordBot] Loading model: {checkpoint_name}")
        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )

        model, clip, vae = out[:3]

        # Cache for next use
        KrakenDiscordBot._cached_model = model
        KrakenDiscordBot._cached_model_name = checkpoint_name
        KrakenDiscordBot._cached_clip = clip
        KrakenDiscordBot._cached_vae = vae

        return model, clip, vae

    def _apply_lora(self, model, clip, lora_name: str, strength: float, clip_strength: float, use_triggers: bool) -> Tuple[Any, Any, str]:
        """
        Apply LoRA to model and clip.
        Returns: (model, clip, trigger_words_string)
        """
        if not lora_name or lora_name == "None":
            return model, clip, ""

        print(f"[KrakenDiscordBot] Loading LoRA: {lora_name}")

        trigger_str = ""

        # Fetch triggers and recommended strengths from CivitAI
        if use_triggers:
            api_key = load_civitai_api_key()
            triggers, rec_model_str, rec_clip_str = civitai_fetch_triggers(lora_name, api_key)

            if triggers:
                trigger_str = ", ".join(triggers)
                print(f"[KrakenDiscordBot] Found triggers: {trigger_str[:50]}...")

            # Use recommended strengths if user left defaults
            if strength == 1.0 and rec_model_str != 1.0:
                strength = rec_model_str
                print(f"[KrakenDiscordBot] Using CivitAI recommended model strength: {strength}")
            if clip_strength == 1.0 and rec_clip_str != 1.0:
                clip_strength = rec_clip_str
                print(f"[KrakenDiscordBot] Using CivitAI recommended clip strength: {clip_strength}")

        # Load and apply LoRA
        loader = LoraLoader()
        model, clip = loader.load_lora(model, clip, lora_name, strength, clip_strength)

        print(f"[KrakenDiscordBot] LoRA applied (model={strength}, clip={clip_strength})")

        return model, clip, trigger_str

    def _load_upscale_model(self, model_name: str):
        """Load upscale model, using cache if available."""
        if not model_name or model_name == "None":
            return None

        if (self._cached_upscale_model_name == model_name and
            self._cached_upscale_model is not None):
            print(f"[KrakenDiscordBot] Using cached upscale model: {model_name}")
            return self._cached_upscale_model

        print(f"[KrakenDiscordBot] Loading upscale model: {model_name}")
        model_path = folder_paths.get_full_path("upscale_models", model_name)

        if not model_path:
            print(f"[KrakenDiscordBot] ERROR: Could not find upscale model path for: {model_name}")
            return None

        try:
            if HAS_SPANDREL:
                print(f"[KrakenDiscordBot] Using Spandrel to load: {model_path}")
                upscale_model = ModelLoader().load_from_file(model_path).model.eval()
            elif model_loading:
                print(f"[KrakenDiscordBot] Using model_loading to load: {model_path}")
                sd = comfy.utils.load_torch_file(model_path)
                upscale_model = model_loading.load_state_dict(sd).eval()
            else:
                print("[KrakenDiscordBot] ERROR: No upscale model loader available (spandrel or model_loading)")
                print("[KrakenDiscordBot] Try: pip install spandrel")
                return None

            device = model_management.get_torch_device()
            upscale_model = upscale_model.to(device)

            # Cache for next use
            KrakenDiscordBot._cached_upscale_model = upscale_model
            KrakenDiscordBot._cached_upscale_model_name = model_name

            print(f"[KrakenDiscordBot] Upscale model loaded successfully: {model_name}")
            return upscale_model

        except Exception as e:
            print(f"[KrakenDiscordBot] ERROR loading upscale model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _upscale_image(self, upscale_model, image: torch.Tensor) -> torch.Tensor:
        """Upscale image using the loaded model."""
        if upscale_model is None:
            return image

        print("[KrakenDiscordBot] Upscaling image...")
        device = model_management.get_torch_device()

        # Image comes in as (B, H, W, C), need (B, C, H, W) for model
        if image.dim() == 4:
            # Move to device and convert to BCHW
            img = image.to(device)
            img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
        else:
            img = image.to(device)

        # Ensure float32
        if img.dtype != torch.float32:
            img = img.float()

        with torch.no_grad():
            upscaled = upscale_model(img)

        # Convert back to BHWC
        if upscaled.dim() == 4:
            upscaled = upscaled.permute(0, 2, 3, 1)  # BCHW -> BHWC

        # Clamp values
        upscaled = torch.clamp(upscaled, 0, 1)

        print(f"[KrakenDiscordBot] Upscaled from {image.shape} to {upscaled.shape}")

        return upscaled.cpu()

    def _load_llm(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """Load LLM for prompt enhancement, using cache if available."""
        if not HAS_TRANSFORMERS:
            print("[KrakenDiscordBot] transformers not installed - cannot enhance prompts")
            return None

        if (self._cached_llm_model_name == model_name and
            self._cached_llm_pipeline is not None):
            print(f"[KrakenDiscordBot] Using cached LLM: {model_name}")
            return self._cached_llm_pipeline

        print(f"[KrakenDiscordBot] Loading LLM for prompt enhancement: {model_name}")
        print("[KrakenDiscordBot] This may take a moment on first use (downloading model)...")

        try:
            has_cuda = torch.cuda.is_available()
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if has_cuda else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            )
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

            llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=processor.tokenizer,
                processor=processor,
                device_map="auto",
            )

            # Cache for reuse
            KrakenDiscordBot._cached_llm_pipeline = llm_pipeline
            KrakenDiscordBot._cached_llm_model_name = model_name

            print("[KrakenDiscordBot] LLM loaded successfully")
            return llm_pipeline

        except Exception as e:
            print(f"[KrakenDiscordBot] Failed to load LLM: {e}")
            return None

    def _unload_llm(self):
        """Unload LLM to free VRAM."""
        if self._cached_llm_pipeline is not None:
            print("[KrakenDiscordBot] Unloading LLM to free VRAM...")
            try:
                del KrakenDiscordBot._cached_llm_pipeline
            except:
                pass
            KrakenDiscordBot._cached_llm_pipeline = None
            KrakenDiscordBot._cached_llm_model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _enhance_prompt(self, prompt: str, style: str = "sdxl") -> Tuple[str, str]:
        """
        Enhance a simple prompt using Qwen2-VL-2B.

        Args:
            prompt: The user's simple prompt
            style: Enhancement style ("sdxl", "anime", "instructional")

        Returns:
            Tuple of (enhanced_prompt, original_prompt)
        """
        if not prompt or not prompt.strip():
            return prompt, prompt

        original = prompt.strip()

        llm = self._load_llm()
        if llm is None:
            print("[KrakenDiscordBot] LLM not available, returning original prompt")
            return original, original

        # Style-specific instructions
        style_instructions = {
            "sdxl": (
                "Rewrite the following concept into a single, highly descriptive and visually rich prompt for a modern text-to-image AI. "
                "Focus on composition, lighting, textures, atmosphere, and camera language. "
                "Add vivid details about colors, materials, environment, and mood. "
                "Keep it under 200 words. Do not add commentary or disclaimers. Output only the enhanced prompt."
            ),
            "anime": (
                "Convert the following concept into a comma-separated list of high-signal tags in danbooru/booru style. "
                "Include quality tags like 'masterpiece, best quality, detailed'. "
                "Add relevant style, lighting, and composition tags. Keep it one line. Output only the tags."
            ),
            "instructional": (
                "Express the following concept as one concise but detailed imperative sentence that fully describes the scene. "
                "Include visual details about lighting, composition, and atmosphere. "
                "Avoid extra words and meta text. Output only the description."
            ),
        }

        # Determine style based on input or default to sdxl
        instruction = style_instructions.get(style.lower(), style_instructions["sdxl"])

        try:
            processor = llm.processor

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}\n\nConcept: '{original}'"}
                    ],
                }
            ]

            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], padding=True, return_tensors="pt")
            inputs = inputs.to(llm.device)

            print(f"[KrakenDiscordBot] Enhancing prompt with LLM...")

            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.12,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                }
                output_ids = llm.model.generate(**inputs, **gen_kwargs)

            # Extract generated text (skip the input prompt tokens)
            generated_ids = [
                output_ids[i, inputs.input_ids[i].shape[0]:]
                for i in range(output_ids.shape[0])
            ]
            enhanced = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            # Clean up any remaining artifacts
            enhanced = enhanced.replace("Enhanced prompt:", "").replace("Output:", "").strip()

            print(f"[KrakenDiscordBot] Enhanced: {enhanced[:100]}...")

            # Unload LLM immediately to free VRAM for image generation
            self._unload_llm()

            return enhanced, original

        except Exception as e:
            print(f"[KrakenDiscordBot] Enhancement failed: {e}")
            self._unload_llm()
            return original, original

    def _encode_prompt(self, clip, text: str) -> List:
        """Encode text prompt to conditioning."""
        if not text or not text.strip():
            return []

        encoder = CLIPTextEncode()
        result = encoder.encode(clip, text)
        return result[0]  # Returns (conditioning,)

    def _create_empty_latent(self, width: int, height: int, batch_size: int = 1) -> Dict:
        """Create empty latent tensor."""
        # Ensure dimensions are divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return {"samples": latent}

    def _sample(
        self,
        model,
        positive,
        negative,
        latent,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float
    ) -> Dict:
        """Run the sampling/generation with progress reporting."""
        from comfy.utils import ProgressBar

        # Handle empty negative
        if not negative:
            negative = []

        device = model_management.get_torch_device()
        latent_image = latent["samples"].to(device)

        # Set up noise
        noise = comfy.sample.prepare_noise(latent_image, seed)

        # Create progress bar for ComfyUI UI
        pbar = ProgressBar(steps)

        # Progress callback
        def progress_callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        try:
            # Try the standard approach with defensive kwargs
            samples = comfy.sample.sample(
                model,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=True,
                noise_mask=None,
                seed=seed,
                callback=progress_callback
            )
        except TypeError:
            # Fallback for older ComfyUI versions (without callback)
            try:
                samples = comfy.sample.sample(
                    model,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_image,
                    denoise=denoise,
                    callback=progress_callback
                )
            except TypeError:
                # Final fallback without callback
                samples = comfy.sample.sample(
                    model,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_image,
                    denoise=denoise
                )

        return {"samples": samples}

    def _decode_latent(self, vae, latent: Dict) -> torch.Tensor:
        """Decode latent to image."""
        samples = latent["samples"]
        images = vae.decode(samples)
        return images

    def _tensor_to_bytes(self, tensor: torch.Tensor, format: str = "PNG") -> bytes:
        """Convert image tensor to bytes for Discord."""
        import io

        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor[0]

        # Convert to numpy
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()

        array = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array)

        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)

        return buffer.getvalue()

    def run_bot(
        self,
        discord_token: str,
        checkpoint: str,
        default_negative: str,
        default_steps: int,
        default_cfg: float,
        default_width: int,
        default_height: int,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        lora_1: str = "None",
        lora_1_strength: float = 1.0,
        lora_2: str = "None",
        lora_2_strength: float = 1.0,
        lora_3: str = "None",
        lora_3_strength: float = 1.0,
        use_civitai_triggers: bool = True,
        civitai_api_status: str = "",
        upscale_model: str = "None",
        rate_limit_seconds: int = 30,
        max_queue_size: int = 10,
        allowed_channels: str = "",
    ):
        """Main execution - wait for Discord command and generate."""

        # Import Discord modules
        try:
            from .kraken_discord.config import get_config
            from .kraken_discord.bot import start_bot, get_bot
            from .kraken_discord.utils import create_generation_embed
        except ImportError as e:
            raise RuntimeError(f"Discord modules not found: {e}. Install discord.py")

        # Set up config
        config = get_config()

        # Handle token - if user provides a new one, save it
        # "Token Loaded" means we already have a saved token, so don't overwrite
        if discord_token and discord_token.strip() and discord_token != "Token Loaded" and not config.is_masked(discord_token):
            print(f"[KrakenDiscordBot] Saving Discord token...")
            config.token = discord_token.strip()
            print(f"[KrakenDiscordBot] Token saved!")

        # Save all settings to config (so they persist)
        config.update({
            "default_negative": default_negative,
            "default_steps": default_steps,
            "default_cfg": default_cfg,
            "default_width": default_width,
            "default_height": default_height,
            "rate_limit_seconds": rate_limit_seconds,
            "max_queue_size": max_queue_size,
        })

        # Parse allowed channels
        if allowed_channels.strip():
            try:
                channel_ids = [int(c.strip()) for c in allowed_channels.split(",") if c.strip()]
                config.allowed_channels = channel_ids
            except ValueError:
                pass

        # Check if we have a token
        if not config.token:
            raise RuntimeError("No Discord token configured! Paste your bot token in the 'discord_token' field and re-queue.")

        # Start bot
        bot = get_bot()
        if bot is None or not bot.is_running:
            print("[KrakenDiscordBot] Starting Discord bot...")
            bot = start_bot(config)

        bot.rate_limiter.update_settings(
            cooldown_seconds=rate_limit_seconds,
            max_queue_size=max_queue_size
        )

        # Pre-load upscale model if selected
        upscale_model_loaded = self._load_upscale_model(upscale_model) if upscale_model != "None" else None

        # Wait for command
        print("[KrakenDiscordBot] Waiting for Discord command...")
        context = bot.get_next_request(timeout=None)

        if context is None:
            raise RuntimeError("No command received")

        print(f"[KrakenDiscordBot] Received: !{context.command_type} from {context.user_name}")
        print(f"[KrakenDiscordBot] Prompt: {context.prompt[:80]}...")

        start_time = time.time()

        try:
            # Load model
            model, clip, vae = self._load_checkpoint(checkpoint)

            # Apply all 3 LoRAs if selected
            all_triggers = []
            lora_names_used = []
            lora_slots = [
                (lora_1, lora_1_strength),
                (lora_2, lora_2_strength),
                (lora_3, lora_3_strength),
            ]

            for lora_name, strength in lora_slots:
                if lora_name and lora_name != "None":
                    model, clip, triggers = self._apply_lora(
                        model, clip, lora_name, strength, strength, use_civitai_triggers
                    )
                    if triggers:
                        all_triggers.append(triggers)
                    lora_names_used.append(os.path.splitext(os.path.basename(lora_name))[0])

            # Get parameters (use context values or defaults)
            prompt = context.prompt
            original_prompt = prompt  # Store original for Discord reporting

            # Apply LLM enhancement if requested (--enhance flag)
            enhanced_prompt = None
            if hasattr(context, 'enhance') and context.enhance:
                print(f"[KrakenDiscordBot] User requested prompt enhancement via --enhance flag")
                # Determine enhancement style based on selected style preset
                enhance_style = "sdxl"  # Default
                if context.style and context.style.lower() == "anime":
                    enhance_style = "anime"
                enhanced_prompt, original_prompt = self._enhance_prompt(prompt, enhance_style)
                prompt = enhanced_prompt
                print(f"[KrakenDiscordBot] Original: {original_prompt[:60]}...")
                print(f"[KrakenDiscordBot] Enhanced: {prompt[:60]}...")

            # Apply quick modifiers (--cinematic, --fog, --masterpiece, etc.)
            if hasattr(context, 'modifiers') and context.modifiers:
                from .kraken_discord.parsers import CommandParser
                parser = CommandParser()
                prompt = parser.apply_modifiers(prompt, context.modifiers)
                print(f"[KrakenDiscordBot] Applied modifiers {context.modifiers}: {prompt[:80]}...")

            # Prepend trigger words if we have them
            if all_triggers:
                trigger_str = ", ".join(all_triggers)
                prompt = f"{trigger_str}, {prompt}"
                print(f"[KrakenDiscordBot] Final prompt with triggers: {prompt[:100]}...")

            negative = context.negative_prompt or default_negative
            width = context.width if context.width else default_width
            height = context.height if context.height else default_height
            steps = context.steps if context.steps else default_steps
            cfg = context.cfg if context.cfg else default_cfg
            seed = context.seed if context.seed and context.seed > 0 else random.randint(0, 2**32 - 1)

            # Ensure dimensions are valid
            width = max(512, min(2048, (width // 8) * 8))
            height = max(512, min(2048, (height // 8) * 8))

            print(f"[KrakenDiscordBot] Generating {width}x{height}, {steps} steps, cfg={cfg}, seed={seed}")

            # Encode prompts
            print("[KrakenDiscordBot] Encoding prompts...")
            positive_cond = self._encode_prompt(clip, prompt)
            negative_cond = self._encode_prompt(clip, negative)

            # Create latent
            latent = self._create_empty_latent(width, height)

            # Sample
            print("[KrakenDiscordBot] Sampling...")
            sampled = self._sample(
                model=model,
                positive=positive_cond,
                negative=negative_cond,
                latent=latent,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )

            # Decode
            print("[KrakenDiscordBot] Decoding...")
            images = self._decode_latent(vae, sampled)

            # Upscale if user requested it via --upscale flag AND model is loaded
            did_upscale = False
            print(f"[KrakenDiscordBot] Upscale check: context.upscale={getattr(context, 'upscale', 'N/A')}, model_loaded={upscale_model_loaded is not None}")
            if hasattr(context, 'upscale') and context.upscale:
                if upscale_model_loaded is not None:
                    print("[KrakenDiscordBot] User requested upscale via --upscale flag - UPSCALING NOW")
                    images = self._upscale_image(upscale_model_loaded, images)
                    did_upscale = True
                    print(f"[KrakenDiscordBot] Upscale complete, did_upscale={did_upscale}")
                else:
                    print("[KrakenDiscordBot] User requested --upscale but no upscale model configured in node")

            gen_time = time.time() - start_time
            print(f"[KrakenDiscordBot] Generated in {gen_time:.1f}s")

            # Determine final size for embed
            if images.dim() == 4:
                final_h, final_w = images.shape[1], images.shape[2]
            else:
                final_h, final_w = images.shape[0], images.shape[1]

            # Create embed
            embed_data = create_generation_embed(
                prompt=prompt,
                negative=negative,
                seed=seed,
                steps=steps,
                cfg=cfg,
                size=f"{final_w}x{final_h}",
                style=context.style or "none",
                model=checkpoint,
                generation_time=gen_time,
                user_name=context.user_name
            )

            # Add LoRA info to embed if used
            if lora_names_used:
                embed_data["fields"] = embed_data.get("fields", [])
                embed_data["fields"].append({
                    "name": "LoRAs",
                    "value": ", ".join(lora_names_used),
                    "inline": True
                })

            # Add enhancement info to embed (show before/after for testing)
            if enhanced_prompt is not None:
                embed_data["fields"] = embed_data.get("fields", [])
                # Truncate long prompts for display
                orig_display = original_prompt[:200] + "..." if len(original_prompt) > 200 else original_prompt
                enh_display = enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt
                embed_data["fields"].append({
                    "name": "Original Prompt",
                    "value": f"```{orig_display}```",
                    "inline": False
                })
                embed_data["fields"].append({
                    "name": "Enhanced Prompt",
                    "value": f"```{enh_display}```",
                    "inline": False
                })

            # Send to Discord
            image_bytes = self._tensor_to_bytes(images)
            success = bot.send_image(
                context=context,
                image_bytes=image_bytes,
                filename="kraken_generated.png",
                embed_data=embed_data
            )

            if success:
                print(f"[KrakenDiscordBot] Sent to {context.user_name}")
            else:
                print("[KrakenDiscordBot] Failed to send")

            return (images,)

        except Exception as e:
            error_msg = str(e)
            print(f"[KrakenDiscordBot] Error: {error_msg}")
            bot.send_error(context, f"Generation failed: {error_msg}")

            # Return empty image on error
            empty = torch.zeros((1, 64, 64, 3))
            return (empty,)


class KrakenDiscordBotStatus:
    """
    Check Discord Bot Status
    Simple utility to see if the bot is running and queue status.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "get_status"
    CATEGORY = "Kraken/Discord"

    def get_status(self):
        try:
            from .kraken_discord.bot import get_bot
            from .kraken_discord.config import get_config

            config = get_config()
            bot = get_bot()

            lines = ["Kraken Discord Bot Status", ""]

            if bot and bot.is_running:
                lines.append("Bot is RUNNING")
                lines.append(f"  Queue: {bot.queue_size}/{bot.rate_limiter.max_queue_size}")
                lines.append(f"  Rate limit: {bot.rate_limiter.cooldown_seconds}s")
            else:
                lines.append("Bot is NOT RUNNING")

            token_status = "SET" if config.token else "NOT SET"
            lines.append(f"  Token: {token_status}")

            if config.allowed_channels:
                lines.append(f"  Allowed channels: {len(config.allowed_channels)}")
            else:
                lines.append("  Allowed channels: ALL")

            return ("\n".join(lines),)

        except Exception as e:
            return (f"Error: {e}",)
