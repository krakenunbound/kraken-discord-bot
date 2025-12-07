"""
Kraken Discord Bot
Core Discord client with async queue management
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List
from enum import Enum

try:
    import discord
    from discord import Message, File, Embed
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None

from .config import get_config, KrakenDiscordConfig
from .parsers import CommandParser, ParsedCommand
from .presets import StylePresets
from .rate_limiter import RateLimiter
from .utils import image_to_bytes, create_generation_embed, format_error_message


class CommandType(Enum):
    """Types of commands the bot handles."""
    GENERATE = "generate"
    IMG2IMG = "img2img"
    HELP = "help"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass
class QueuedRequest:
    """A request waiting in the queue."""
    command: ParsedCommand
    message: Any  # Discord message object
    added_time: float = field(default_factory=time.time)
    position: int = 0

    # Response callbacks (set by bot)
    reply_callback: Optional[Callable] = None
    edit_callback: Optional[Callable] = None


@dataclass
class DiscordContext:
    """
    Context passed through ComfyUI workflow for responding to Discord.
    This is the DISCORD_CONTEXT output type.
    """
    request: QueuedRequest
    bot: 'KrakenDiscordBot'

    # Computed values for workflow
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    seed: int = -1
    style: str = "none"
    model: str = ""
    command_type: str = "generate"

    # Image input (for img2img)
    has_input_image: bool = False
    input_image_url: Optional[str] = None

    # User info
    user_id: int = 0
    user_name: str = ""
    channel_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg": self.cfg,
            "seed": self.seed,
            "style": self.style,
            "model": self.model,
            "command_type": self.command_type,
            "has_input_image": self.has_input_image,
            "input_image_url": self.input_image_url,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "channel_id": self.channel_id,
        }


class KrakenDiscordBot:
    """
    Discord bot for image generation.
    Runs in a background thread and queues requests for ComfyUI.
    """

    def __init__(self, config: Optional[KrakenDiscordConfig] = None):
        """
        Initialize the bot.

        Args:
            config: Configuration object (uses global config if not provided)
        """
        if not DISCORD_AVAILABLE:
            raise ImportError("discord.py is required. Install with: pip install discord.py")

        self.config = config or get_config()
        self.parser = CommandParser(prefix=self.config.get("command_prefix", "!"))
        self.rate_limiter = RateLimiter(
            cooldown_seconds=self.config.get("rate_limit_seconds", 30),
            max_queue_size=self.config.get("max_queue_size", 10)
        )

        # Queue for pending requests
        self._queue: deque = deque()
        self._queue_lock = threading.Lock()
        self._data_ready = threading.Event()

        # Discord client
        self._client: Optional[discord.Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Current request being processed
        self._current_request: Optional[QueuedRequest] = None

    def start(self) -> bool:
        """
        Start the Discord bot in a background thread.

        Returns:
            True if started successfully
        """
        token = self.config.token
        if not token:
            print("[KrakenDiscord] No Discord token configured!")
            return False

        if self._running:
            print("[KrakenDiscord] Bot already running")
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run_bot, daemon=True)
        self._thread.start()

        # Wait briefly for bot to start
        time.sleep(1)
        return True

    def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run_bot(self) -> None:
        """Run the Discord bot (called in background thread)."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        intents = discord.Intents.default()
        intents.message_content = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            print(f"[KrakenDiscord] Bot connected as {self._client.user}")
            status = self.config.get("bot_status", "Generating images...")
            await self._client.change_presence(
                activity=discord.Activity(type=discord.ActivityType.playing, name=status)
            )

        @self._client.event
        async def on_message(message: Message):
            await self._handle_message(message)

        try:
            self._loop.run_until_complete(self._client.start(self.config.token))
        except Exception as e:
            print(f"[KrakenDiscord] Bot error: {e}")
        finally:
            self._running = False

    async def _handle_message(self, message: Message) -> None:
        """Handle incoming Discord message."""
        # Ignore own messages
        if message.author == self._client.user:
            return

        # Check if it's a command
        content = message.content.strip()
        prefix = self.config.get("command_prefix", "!")
        if not content.startswith(prefix):
            return

        # Check channel permissions
        if not self.config.is_channel_allowed(message.channel.id):
            return

        # Parse command
        parsed = self.parser.parse(content, message.attachments)
        if not parsed:
            return

        # Add user info
        parsed.user_id = message.author.id
        parsed.user_name = message.author.display_name
        parsed.channel_id = message.channel.id
        parsed.message_id = message.id

        # Route command
        cmd_type = self._get_command_type(parsed.command)

        if cmd_type == CommandType.HELP:
            await self._send_help(message)
            return

        if cmd_type == CommandType.STATUS:
            await self._send_status(message)
            return

        if cmd_type == CommandType.UNKNOWN:
            await message.reply(f"Unknown command: `{parsed.command}`. Use `{prefix}help` for available commands.")
            return

        # Check for img2img without image
        if cmd_type == CommandType.IMG2IMG and not parsed.has_image:
            await message.reply("**Error:** `!img2img` requires an attached image. Please attach an image to your message.")
            return

        # Check rate limit
        rate_info = self.rate_limiter.check(parsed.user_id)
        if rate_info.is_limited:
            await message.reply(rate_info.message)
            return

        # Check queue
        queue_full, queue_count, queue_msg = self.rate_limiter.check_queue()
        if queue_full:
            await message.reply(queue_msg)
            return

        # Add to queue
        position = self.rate_limiter.add_to_queue()
        self.rate_limiter.record_request(parsed.user_id)

        request = QueuedRequest(
            command=parsed,
            message=message,
            position=position
        )

        # Set up callbacks
        async def reply_callback(content=None, embed=None, file=None):
            try:
                await message.reply(content=content, embed=embed, file=file)
            except Exception as e:
                print(f"[KrakenDiscord] Reply error: {e}")

        async def edit_callback(content):
            try:
                # Can't edit original message, but we can send follow-up
                pass
            except Exception as e:
                print(f"[KrakenDiscord] Edit error: {e}")

        request.reply_callback = reply_callback
        request.edit_callback = edit_callback

        # Build acknowledgment message with details
        ack_lines = []

        # Show what we're generating
        prompt_preview = parsed.prompt[:100] + "..." if len(parsed.prompt) > 100 else parsed.prompt
        ack_lines.append(f"**Generating:** {prompt_preview}")

        # Show settings if customized
        settings_parts = []
        if parsed.steps:
            settings_parts.append(f"Steps: {parsed.steps}")
        if parsed.cfg:
            settings_parts.append(f"CFG: {parsed.cfg}")
        if parsed.width or parsed.height:
            w = parsed.width or 1024
            h = parsed.height or 1024
            settings_parts.append(f"Size: {w}x{h}")
        if parsed.style:
            settings_parts.append(f"Style: {parsed.style}")
        if parsed.seed and parsed.seed > 0:
            settings_parts.append(f"Seed: {parsed.seed}")
        if parsed.negative:
            neg_preview = parsed.negative[:50] + "..." if len(parsed.negative) > 50 else parsed.negative
            settings_parts.append(f"Negative: {neg_preview}")

        if settings_parts:
            ack_lines.append("**Settings:** " + " | ".join(settings_parts))

        # Show queue position
        if position > 1:
            ack_lines.append(f"**Queue Position:** #{position} (estimated wait: ~{(position-1) * 30}s)")
        else:
            ack_lines.append("**Status:** Processing now...")

        # Send acknowledgment
        ack_message = "\n".join(ack_lines)
        await message.reply(ack_message)

        # Add to processing queue
        with self._queue_lock:
            self._queue.append(request)
            self._data_ready.set()

    def _get_command_type(self, command: str) -> CommandType:
        """Get the type of command."""
        cmd = command.lower()
        if cmd in ("generate", "gen", "g", "create", "make"):
            return CommandType.GENERATE
        elif cmd in ("img2img", "i2i", "transform"):
            return CommandType.IMG2IMG
        elif cmd in ("help", "h", "?", "commands"):
            return CommandType.HELP
        elif cmd in ("status", "queue", "q"):
            return CommandType.STATUS
        else:
            return CommandType.UNKNOWN

    async def _send_help(self, message: Message) -> None:
        """Send help message."""
        help_text = self.parser.get_help_text()
        await message.reply(help_text)

    async def _send_status(self, message: Message) -> None:
        """Send status message."""
        queue_count = self.rate_limiter.queue_count
        status = f"**Bot Status**\nQueue: {queue_count}/{self.rate_limiter.max_queue_size}"
        await message.reply(status)

    def get_next_request(self, timeout: float = None) -> Optional[DiscordContext]:
        """
        Get the next request from the queue (blocking).
        Called by ComfyUI node.

        Args:
            timeout: Maximum time to wait (None = forever)

        Returns:
            DiscordContext or None if timeout
        """
        # Wait for data
        if not self._data_ready.wait(timeout=timeout):
            return None

        with self._queue_lock:
            if not self._queue:
                self._data_ready.clear()
                return None

            request = self._queue.popleft()
            if not self._queue:
                self._data_ready.clear()

        self._current_request = request

        # Build context with defaults
        parsed = request.command
        config = self.config

        # Get default values
        default_steps = config.get("default_steps", 20)
        default_cfg = config.get("default_cfg", 7.0)
        default_width = config.get("default_width", 1024)
        default_height = config.get("default_height", 1024)
        default_style = config.get("default_style", "none")
        default_model = config.get("default_model", "")

        # Apply user overrides
        steps = parsed.steps if parsed.steps is not None else default_steps
        cfg = parsed.cfg if parsed.cfg is not None else default_cfg
        width = parsed.width if parsed.width is not None else default_width
        height = parsed.height if parsed.height is not None else default_height
        style = parsed.style if parsed.style is not None else default_style
        model = parsed.model if parsed.model is not None else default_model

        # Apply style preset
        prompt, negative, cfg, steps = StylePresets.apply_style(
            parsed.prompt,
            parsed.negative,
            style,
            cfg,
            steps
        )

        # Determine command type
        cmd_type = self._get_command_type(parsed.command)

        context = DiscordContext(
            request=request,
            bot=self,
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=parsed.seed if parsed.seed is not None else -1,
            style=style,
            model=model,
            command_type=cmd_type.value,
            has_input_image=parsed.has_image,
            input_image_url=parsed.image_url,
            user_id=parsed.user_id,
            user_name=parsed.user_name,
            channel_id=parsed.channel_id,
        )

        return context

    def send_image(
        self,
        context: DiscordContext,
        image_bytes: bytes,
        filename: str = "generated.png",
        embed_data: Optional[Dict] = None
    ) -> bool:
        """
        Send generated image back to Discord.

        Args:
            context: The DiscordContext from get_next_request
            image_bytes: PNG/WebP image bytes
            filename: Filename for the attachment
            embed_data: Optional embed dictionary

        Returns:
            True if sent successfully
        """
        if not self._loop or not context.request.reply_callback:
            return False

        try:
            import io
            file = discord.File(io.BytesIO(image_bytes), filename=filename)
            embed = Embed.from_dict(embed_data) if embed_data else None

            future = asyncio.run_coroutine_threadsafe(
                context.request.reply_callback(file=file, embed=embed),
                self._loop
            )
            future.result(timeout=30)

            # Remove reaction and update queue
            try:
                asyncio.run_coroutine_threadsafe(
                    context.request.message.remove_reaction("hourglass", self._client.user),
                    self._loop
                ).result(timeout=5)
            except:
                pass

            self.rate_limiter.remove_from_queue()
            return True

        except Exception as e:
            print(f"[KrakenDiscord] Failed to send image: {e}")
            return False

    def send_text(self, context: DiscordContext, text: str) -> bool:
        """
        Send text message back to Discord.

        Args:
            context: The DiscordContext
            text: Message text

        Returns:
            True if sent successfully
        """
        if not self._loop or not context.request.reply_callback:
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                context.request.reply_callback(content=text),
                self._loop
            )
            future.result(timeout=30)
            return True
        except Exception as e:
            print(f"[KrakenDiscord] Failed to send text: {e}")
            return False

    def send_error(self, context: DiscordContext, error: str) -> bool:
        """
        Send error message and clean up queue.

        Args:
            context: The DiscordContext
            error: Error message

        Returns:
            True if sent successfully
        """
        self.rate_limiter.remove_from_queue()
        return self.send_text(context, format_error_message(error))

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running and self._client is not None

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        with self._queue_lock:
            return len(self._queue)


# Singleton bot instance
_bot_instance: Optional[KrakenDiscordBot] = None


def get_bot() -> Optional[KrakenDiscordBot]:
    """Get the global bot instance."""
    return _bot_instance


def start_bot(config: Optional[KrakenDiscordConfig] = None) -> KrakenDiscordBot:
    """
    Start or get the global bot instance.

    Args:
        config: Configuration (uses global config if not provided)

    Returns:
        KrakenDiscordBot instance
    """
    global _bot_instance

    if _bot_instance is None or not _bot_instance.is_running:
        _bot_instance = KrakenDiscordBot(config)
        _bot_instance.start()

    return _bot_instance


def stop_bot() -> None:
    """Stop the global bot instance."""
    global _bot_instance
    if _bot_instance:
        _bot_instance.stop()
        _bot_instance = None
