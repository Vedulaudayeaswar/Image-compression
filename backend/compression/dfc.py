from __future__ import annotations

import hashlib
import json
import struct
from typing import Any


MAGIC = b"DFC1"


def serialize(metadata: dict[str, Any], entropy_payload: bytes) -> bytes:
    """DFC v1: magic | JSON metadata length | metadata | entropy payload | SHA-256."""
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = MAGIC + struct.pack("<I", len(header)) + header + entropy_payload
    return body + hashlib.sha256(body).digest()


def deserialize(blob: bytes) -> tuple[dict[str, Any], bytes]:
    if len(blob) < 40 or blob[:4] != MAGIC:
        raise ValueError("Not a valid DFC v1 file.")
    body, checksum = blob[:-32], blob[-32:]
    if hashlib.sha256(body).digest() != checksum:
        raise ValueError("DFC checksum verification failed.")
    length = struct.unpack_from("<I", blob, 4)[0]
    end = 8 + length
    if end > len(body): raise ValueError("Malformed DFC metadata.")
    metadata = json.loads(blob[8:end].decode("utf-8"))
    if metadata.get("version") != 1: raise ValueError("Unsupported DFC version.")
    return metadata, blob[end:-32]
