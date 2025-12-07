"""
Kraken Discord Configuration
Handles token storage, loading, and masking for secure display
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


class KrakenDiscordConfig:
    """
    Configuration manager for Kraken Discord bot.
    Supports config file storage with masked token display in workflows.
    """

    CONFIG_FILENAME = "kraken_discord_config.json"

    # Default configuration values
    DEFAULTS = {
        "discord_token": "",
        "command_prefix": "!",
        "default_steps": 20,
        "default_cfg": 7.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_model": "",
        "default_style": "none",
        "default_negative": "ugly, blurry, low quality, distorted",
        "rate_limit_seconds": 30,
        "max_queue_size": 10,
        "allowed_channels": [],  # Empty = all channels allowed
        "admin_user_ids": [],    # Discord user IDs with admin access
        "enable_img2img": True,
        "enable_help": True,
        "bot_status": "Generating images...",
    }

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.

        Args:
            config_dir: Directory to store config file. Defaults to kraken_discord folder.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self.config_path = self.config_dir / self.CONFIG_FILENAME
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file, creating with defaults if needed."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in self.DEFAULTS.items():
                    if key not in self._config:
                        self._config[key] = value
            except (json.JSONDecodeError, IOError) as e:
                print(f"[KrakenDiscord] Error loading config: {e}")
                self._config = self.DEFAULTS.copy()
        else:
            self._config = self.DEFAULTS.copy()
            self._save_config()

    def _save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"[KrakenDiscord] Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default if default is not None else self.DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and save."""
        self._config[key] = value
        self._save_config()

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values at once."""
        self._config.update(updates)
        self._save_config()

    @property
    def token(self) -> str:
        """Get the Discord bot token."""
        return self._config.get("discord_token", "")

    @token.setter
    def token(self, value: str) -> None:
        """Set the Discord bot token."""
        self.set("discord_token", value)

    @staticmethod
    def mask_token(token: str) -> str:
        """
        Mask a token for display in UI.
        Shows first 4 and last 4 characters only.

        Args:
            token: The full token string

        Returns:
            Masked token like "MTIz****...****abcd" or empty string
        """
        if not token:
            return ""
        if len(token) <= 12:
            return "*" * len(token)
        return f"{token[:4]}{'*' * 20}...{'*' * 4}{token[-4:]}"

    @staticmethod
    def is_masked(value: str) -> bool:
        """Check if a value appears to be a masked token."""
        if not value:
            return False
        return "****" in value or value == "*" * len(value)

    def get_display_token(self) -> str:
        """Get masked token for UI display."""
        return self.mask_token(self.token)

    def set_token_if_not_masked(self, value: str) -> bool:
        """
        Set token only if the value isn't masked.
        Returns True if token was updated, False if masked value was ignored.

        This allows the workflow to display masked tokens while still
        allowing users to paste in new tokens.
        """
        if self.is_masked(value):
            return False
        if value and value != self.token:
            self.token = value
            return True
        return False

    @property
    def allowed_channels(self) -> List[int]:
        """Get list of allowed channel IDs."""
        channels = self._config.get("allowed_channels", [])
        # Convert string IDs to integers if needed
        return [int(c) for c in channels if c]

    @allowed_channels.setter
    def allowed_channels(self, value: List[int]) -> None:
        """Set allowed channel IDs."""
        self.set("allowed_channels", value)

    @property
    def admin_user_ids(self) -> List[int]:
        """Get list of admin user IDs."""
        admins = self._config.get("admin_user_ids", [])
        return [int(a) for a in admins if a]

    def is_channel_allowed(self, channel_id: int) -> bool:
        """Check if a channel is allowed for bot commands."""
        allowed = self.allowed_channels
        if not allowed:  # Empty list = all channels allowed
            return True
        return channel_id in allowed

    def is_admin(self, user_id: int) -> bool:
        """Check if a user has admin privileges."""
        return user_id in self.admin_user_ids

    def to_dict(self) -> Dict[str, Any]:
        """Get full config as dictionary (token masked)."""
        result = self._config.copy()
        result["discord_token"] = self.get_display_token()
        return result

    def __repr__(self) -> str:
        return f"KrakenDiscordConfig(path={self.config_path}, token={'SET' if self.token else 'NOT SET'})"


# Singleton instance for easy access
_config_instance: Optional[KrakenDiscordConfig] = None


def get_config() -> KrakenDiscordConfig:
    """Get the global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = KrakenDiscordConfig()
    return _config_instance


def reload_config() -> KrakenDiscordConfig:
    """Force reload of configuration from disk."""
    global _config_instance
    _config_instance = KrakenDiscordConfig()
    return _config_instance
