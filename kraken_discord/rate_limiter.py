"""
Kraken Discord Rate Limiter
Per-user rate limiting to prevent spam
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from threading import Lock


@dataclass
class RateLimitInfo:
    """Information about a user's rate limit status."""
    is_limited: bool
    remaining_seconds: float
    message: str


class RateLimiter:
    """
    Per-user rate limiter with configurable cooldown.
    Thread-safe for use with async Discord bot.
    """

    def __init__(self, cooldown_seconds: float = 30.0, max_queue_size: int = 10):
        """
        Initialize rate limiter.

        Args:
            cooldown_seconds: Seconds between allowed requests per user
            max_queue_size: Maximum pending requests in queue
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_queue_size = max_queue_size

        # Track last request time per user
        self._last_request: Dict[int, float] = defaultdict(float)

        # Track current queue
        self._queue_count = 0

        # Thread safety
        self._lock = Lock()

    def check(self, user_id: int) -> RateLimitInfo:
        """
        Check if a user is rate limited.

        Args:
            user_id: Discord user ID

        Returns:
            RateLimitInfo with status and remaining time
        """
        with self._lock:
            now = time.time()
            last = self._last_request[user_id]
            elapsed = now - last

            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                return RateLimitInfo(
                    is_limited=True,
                    remaining_seconds=remaining,
                    message=f"Please wait {remaining:.0f} seconds before generating again."
                )

            return RateLimitInfo(
                is_limited=False,
                remaining_seconds=0,
                message=""
            )

    def record_request(self, user_id: int) -> None:
        """
        Record that a user made a request.

        Args:
            user_id: Discord user ID
        """
        with self._lock:
            self._last_request[user_id] = time.time()

    def get_remaining_time(self, user_id: int) -> float:
        """
        Get remaining cooldown time for a user.

        Args:
            user_id: Discord user ID

        Returns:
            Seconds remaining (0 if not limited)
        """
        with self._lock:
            now = time.time()
            last = self._last_request[user_id]
            elapsed = now - last

            if elapsed < self.cooldown_seconds:
                return self.cooldown_seconds - elapsed
            return 0

    def reset_user(self, user_id: int) -> None:
        """
        Reset rate limit for a specific user.

        Args:
            user_id: Discord user ID
        """
        with self._lock:
            self._last_request[user_id] = 0

    def reset_all(self) -> None:
        """Reset all rate limits."""
        with self._lock:
            self._last_request.clear()

    # Queue management

    def check_queue(self) -> Tuple[bool, int, str]:
        """
        Check if queue has room for new request.

        Returns:
            Tuple of (is_full, current_count, message)
        """
        with self._lock:
            if self._queue_count >= self.max_queue_size:
                return True, self._queue_count, f"Queue is full ({self._queue_count}/{self.max_queue_size}). Please try again later."
            return False, self._queue_count, ""

    def add_to_queue(self) -> int:
        """
        Add request to queue.

        Returns:
            Position in queue (1-indexed)
        """
        with self._lock:
            self._queue_count += 1
            return self._queue_count

    def remove_from_queue(self) -> int:
        """
        Remove request from queue.

        Returns:
            New queue count
        """
        with self._lock:
            self._queue_count = max(0, self._queue_count - 1)
            return self._queue_count

    def get_queue_position(self, position: int) -> str:
        """
        Get human-readable queue position message.

        Args:
            position: Position in queue (1-indexed)

        Returns:
            Status message
        """
        if position == 1:
            return "Your request is being processed..."
        else:
            wait_estimate = (position - 1) * 30  # Rough estimate based on ~30s per image
            return f"You are #{position} in queue (estimated wait: ~{wait_estimate}s)"

    @property
    def queue_count(self) -> int:
        """Get current queue count."""
        with self._lock:
            return self._queue_count

    def update_settings(self, cooldown_seconds: Optional[float] = None, max_queue_size: Optional[int] = None) -> None:
        """
        Update rate limiter settings.

        Args:
            cooldown_seconds: New cooldown (if provided)
            max_queue_size: New max queue size (if provided)
        """
        with self._lock:
            if cooldown_seconds is not None:
                self.cooldown_seconds = cooldown_seconds
            if max_queue_size is not None:
                self.max_queue_size = max_queue_size
