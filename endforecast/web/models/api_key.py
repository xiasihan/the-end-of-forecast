"""API key model for production prediction auth."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ApiKey:
    id: str
    name: str
    key_hash: str  # SHA256 of the full key
    prefix: str     # First 8 chars for display
    created_at: str = ""
    last_used: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def generate(cls, name: str) -> tuple["ApiKey", str]:
        """Generate a new API key. Returns (ApiKey, raw_key_string)."""
        raw = "ef_" + secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw.encode()).hexdigest()
        prefix = raw[:10]
        key_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return cls(id=key_id, name=name, key_hash=key_hash, prefix=prefix), raw

    def verify(self, raw_key: str) -> bool:
        return hashlib.sha256(raw_key.encode()).hexdigest() == self.key_hash