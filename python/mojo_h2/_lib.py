"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

from ._tables import HUFFMAN_CODES, HUFFMAN_CODE_LENGTHS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "h2_kernels.mojo")
LIB = os.environ.get("MOJO_H2_LIB") or os.path.join(
    ROOT, "dist", "libmojo-h2.so"
)

I = ctypes.c_int64
_SIGNATURES = {
    "mh2_huffman_encoded_size": ([I, I, I], I),
    "mh2_huffman_encode": ([I, I, I, I, I], I),
    "mh2_huffman_decode": ([I, I, I, I, I, I, I, I], I),
    "mh2_encode_integer": ([I, I, I], I),
    "mh2_decode_integer": ([I, I, I, I], I),
    "mh2_encode_frame_headers": ([I, I, I, I, I, I], None),
    "mh2_decode_frame_headers": ([I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_H2_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_H2_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_H2_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.exists(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo source at {SRC} and no shared library at {LIB}")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(SRC):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build", "--emit", "shared-lib", SRC, "-o", LIB,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_lib, name)
            function.argtypes = argtypes
            function.restype = restype
    return _lib


def _addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and not address:
        raise ValueError("non-empty NumPy buffer has a null address")
    return address


def _byte_array(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Return a contiguous, one-dimensional byte view kept alive by its caller."""
    view = memoryview(data)
    if not view.c_contiguous:
        raise BufferError("buffer must be C-contiguous")
    try:
        byte_view = view.cast("B")
    except TypeError as error:
        raise BufferError("buffer must be byte-addressable") from error
    return np.frombuffer(byte_view, dtype=np.uint8)


def _integer_array(values, name: str, minimum: int, maximum: int, dtype) -> np.ndarray:
    """Validate before casting so NumPy cannot silently truncate or wrap values."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not raw.size:
        return np.empty(0, dtype=dtype)
    if raw.dtype.kind not in "iub":
        raise TypeError(f"{name} must contain integers")
    if raw.size and (np.any(raw < minimum) or np.any(raw > maximum)):
        raise ValueError(f"{name} values must be between {minimum} and {maximum}")
    return np.ascontiguousarray(raw, dtype=dtype)


_CODES = np.asarray(HUFFMAN_CODES, dtype=np.uint32)
_LENGTHS = np.asarray(HUFFMAN_CODE_LENGTHS, dtype=np.uint8)


def _decode_tree() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = [-1]
    right = [-1]
    symbols = [-1]
    for symbol, (code, width) in enumerate(zip(HUFFMAN_CODES, HUFFMAN_CODE_LENGTHS)):
        node = 0
        for shift in range(width - 1, -1, -1):
            branch = right if (code >> shift) & 1 else left
            child = branch[node]
            if child < 0:
                child = len(left)
                branch[node] = child
                left.append(-1)
                right.append(-1)
                symbols.append(-1)
            node = child
        symbols[node] = symbol
    accepting = [0] * len(left)
    node = 0
    for _ in range(7):
        node = right[node]
        accepting[node] = 1
    return (
        np.asarray(left, dtype=np.int32),
        np.asarray(right, dtype=np.int32),
        np.asarray(symbols, dtype=np.int32),
        np.asarray(accepting, dtype=np.uint8),
    )


_LEFT, _RIGHT, _SYMBOLS, _ACCEPTING = _decode_tree()


def huffman_encode(data: bytes) -> bytes:
    if not data:
        return b""
    source = _byte_array(data)
    size = lib().mh2_huffman_encoded_size(_addr(source), source.size, _addr(_LENGTHS))
    if size < 0:
        raise RuntimeError("Mojo Huffman sizing failed")
    destination = np.empty(size, dtype=np.uint8)
    written = lib().mh2_huffman_encode(
        _addr(source), source.size, _addr(_CODES), _addr(_LENGTHS), _addr(destination)
    )
    if written < 0 or written > destination.size:
        raise RuntimeError("Mojo Huffman encoder returned an invalid output length")
    return destination[:written].tobytes()


def huffman_decode(data: bytes) -> bytes:
    if not data:
        return b""
    source = _byte_array(data)
    destination = np.empty(source.size * 2 + 1, dtype=np.uint8)
    written = lib().mh2_huffman_decode(
        _addr(source), source.size, _addr(_LEFT), _addr(_RIGHT),
        _addr(_SYMBOLS), _addr(_ACCEPTING), _addr(destination), destination.size,
    )
    if written < 0:
        raise ValueError("Invalid Huffman string")
    return destination[:written].tobytes()


def encode_integer_mojo(value: int, prefix_bits: int) -> bytearray:
    if value > np.iinfo(np.int64).max:
        raise OverflowError("HPACK integer exceeds the Mojo signed 64-bit limit")
    destination = np.empty(16, dtype=np.uint8)
    written = lib().mh2_encode_integer(value, prefix_bits, _addr(destination))
    if written < 1 or written > destination.size:
        raise RuntimeError("Mojo integer encoder returned an invalid output length")
    return bytearray(destination[:written])


def decode_integer_mojo(data: bytes | memoryview, prefix_bits: int) -> tuple[int, int]:
    source = _byte_array(data)
    if not source.size:
        raise ValueError("Invalid HPACK integer")
    value = np.empty(1, dtype=np.int64)
    consumed = lib().mh2_decode_integer(
        _addr(source), source.size, prefix_bits, _addr(value)
    )
    if consumed < 0:
        raise ValueError("Invalid HPACK integer")
    return int(value[0]), consumed


def encode_frame_headers(
    lengths, types, flags, stream_ids,
) -> bytes:
    lengths_array = _integer_array(
        lengths, "lengths", 0, 0xFFFFFF, np.int64
    )
    types_array = _integer_array(types, "types", 0, 0xFF, np.uint8)
    flags_array = _integer_array(flags, "flags", 0, 0xFF, np.uint8)
    streams_array = _integer_array(
        stream_ids, "stream_ids", 0, 0xFFFFFFFF, np.int64
    )
    n = lengths_array.size
    if not (types_array.size == flags_array.size == streams_array.size == n):
        raise ValueError("all frame-header arrays must have the same length")
    if not n:
        return b""
    destination = np.empty(n * 9, dtype=np.uint8)
    lib().mh2_encode_frame_headers(
        _addr(lengths_array), _addr(types_array), _addr(flags_array),
        _addr(streams_array), _addr(destination), n,
    )
    return destination.tobytes()


def decode_frame_headers(data: bytes | bytearray | memoryview):
    source = _byte_array(data)
    if source.size % 9:
        raise ValueError("frame-header buffer length must be a multiple of 9")
    n = source.size // 9
    lengths = np.empty(n, dtype=np.int64)
    types = np.empty(n, dtype=np.uint8)
    flags = np.empty(n, dtype=np.uint8)
    streams = np.empty(n, dtype=np.int64)
    if n:
        lib().mh2_decode_frame_headers(
            _addr(source), n, _addr(lengths), _addr(types), _addr(flags), _addr(streams)
        )
    return lengths, types, flags, streams


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
