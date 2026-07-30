"""HPACK encoder and decoder with upstream-compatible public contracts."""

from __future__ import annotations

from collections import deque
from typing import Any

from ._lib import (
    decode_integer_mojo,
    encode_integer_mojo,
    huffman_decode,
    huffman_encode,
)
from ._tables import STATIC_TABLE
from .exceptions import (
    HPACKDecodingError,
    InvalidTableIndex,
    InvalidTableSizeError,
    OversizedHeaderListError,
)

_STATIC_EXACT = {
    entry: index for index, entry in enumerate(STATIC_TABLE, 1)
}
_STATIC_NAMES: dict[bytes, int] = {}
for _index, (_name, _) in enumerate(STATIC_TABLE, 1):
    _STATIC_NAMES.setdefault(_name, _index)
_ONE_BYTE_INDEXED = tuple(bytes((0x80 | index,)) for index in range(127))


class HeaderTuple(tuple):
    __slots__ = ()
    indexable = True

    def __new__(cls, *args):
        return tuple.__new__(cls, args)


class NeverIndexedHeaderTuple(HeaderTuple):
    __slots__ = ()
    indexable = False


def table_entry_size(name: bytes, value: bytes) -> int:
    return 32 + len(name) + len(value)


def encode_integer(integer: int, prefix_bits: int) -> bytearray:
    if integer < 0:
        raise ValueError(f"Can only encode positive integers, got {integer}")
    if not 1 <= prefix_bits <= 8:
        raise ValueError(f"Prefix bits must be between 1 and 8, got {prefix_bits}")
    return encode_integer_mojo(integer, prefix_bits)


def decode_integer(
    data: bytes | memoryview, prefix_bits: int,
) -> tuple[int, int]:
    if not 1 <= prefix_bits <= 8:
        raise ValueError(f"Prefix bits must be between 1 and 8, got {prefix_bits}")
    try:
        return decode_integer_mojo(data, prefix_bits)
    except (ValueError, BufferError) as error:
        raise HPACKDecodingError(
            f"Unable to decode HPACK integer representation from {data!r}"
        ) from error


class HeaderTable:
    DEFAULT_SIZE = 4096
    STATIC_TABLE = STATIC_TABLE
    STATIC_TABLE_LENGTH = len(STATIC_TABLE)

    def __init__(self) -> None:
        self._maxsize = self.DEFAULT_SIZE
        self._current_size = 0
        self.resized = False
        self.dynamic_entries: deque[tuple[bytes, bytes]] = deque()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int) -> None:
        value = int(value)
        old = self._maxsize
        self._maxsize = value
        self.resized = value != old
        if value <= 0:
            self.dynamic_entries.clear()
            self._current_size = 0
        elif value < old:
            self._shrink()

    def _shrink(self) -> None:
        while self._current_size > self._maxsize:
            name, value = self.dynamic_entries.pop()
            self._current_size -= table_entry_size(name, value)

    def add(self, name: bytes, value: bytes) -> None:
        size = table_entry_size(name, value)
        if size > self._maxsize:
            self.dynamic_entries.clear()
            self._current_size = 0
            return
        self.dynamic_entries.appendleft((name, value))
        self._current_size += size
        self._shrink()

    def get_by_index(self, index: int) -> tuple[bytes, bytes]:
        if index <= 0:
            raise InvalidTableIndex(f"Invalid table index {index}")
        if index <= self.STATIC_TABLE_LENGTH:
            return self.STATIC_TABLE[index - 1]
        dynamic_index = index - self.STATIC_TABLE_LENGTH - 1
        try:
            return self.dynamic_entries[dynamic_index]
        except IndexError as error:
            raise InvalidTableIndex(f"Invalid table index {index}") from error

    def search(
        self, name: bytes, value: bytes,
    ) -> tuple[int, bytes, bytes | None] | None:
        exact = _STATIC_EXACT.get((name, value))
        if exact is not None:
            return exact, name, value
        static_name = _STATIC_NAMES.get(name)
        partial = (
            (static_name, name, None) if static_name is not None else None
        )
        offset = self.STATIC_TABLE_LENGTH + 1
        for index, (candidate_name, candidate_value) in enumerate(
            self.dynamic_entries, offset
        ):
            if candidate_name == name:
                if candidate_value == value:
                    return index, name, value
                if partial is None:
                    partial = (index, name, None)
        return partial


def _to_bytes(value: bytes | str | Any) -> bytes:
    if type(value) is bytes:
        return value
    if type(value) is not str:
        value = str(value)
    return value.encode("utf-8")


def _encode_string(value: bytes, use_huffman: bool) -> bytes:
    encoded = huffman_encode(value) if use_huffman else value
    length = encode_integer(len(encoded), 7)
    if use_huffman:
        length[0] |= 0x80
    return bytes(length) + encoded


class Encoder:
    def __init__(self) -> None:
        self.header_table = HeaderTable()
        self.table_size_changes: list[int] = []

    @property
    def header_table_size(self) -> int:
        return self.header_table.maxsize

    @header_table_size.setter
    def header_table_size(self, value: int) -> None:
        self.header_table.maxsize = value
        if self.header_table.resized:
            self.table_size_changes.append(value)

    def encode(self, headers, huffman: bool = True) -> bytes:
        pieces = []
        if self.header_table.resized:
            for size in self.table_size_changes:
                encoded_size = encode_integer(size, 5)
                encoded_size[0] |= 0x20
                pieces.append(bytes(encoded_size))
            self.table_size_changes.clear()
            self.header_table.resized = False

        if isinstance(headers, dict):
            items = sorted(
                headers.items(), key=lambda item: not _to_bytes(item[0]).startswith(b":")
            )
        else:
            items = iter(headers)

        for header in items:
            sensitive = False
            if isinstance(header, HeaderTuple):
                sensitive = not header.indexable
            elif len(header) > 2:
                sensitive = bool(header[2])
            pieces.append(
                self.add(
                    (_to_bytes(header[0]), _to_bytes(header[1])),
                    sensitive,
                    huffman,
                )
            )
        return b"".join(pieces)

    def add(
        self, to_add: tuple[bytes, bytes], sensitive: bool, huffman: bool = False,
    ) -> bytes:
        name, value = to_add
        index_prefix = 0x10 if sensitive else 0x40
        match = self.header_table.search(name, value)
        if match is None:
            encoded = bytes([index_prefix]) + _encode_string(
                name, huffman
            ) + _encode_string(value, huffman)
            if not sensitive:
                self.header_table.add(name, value)
            return encoded

        index, matched_name, perfect = match
        if perfect is not None:
            if index < 127:
                return _ONE_BYTE_INDEXED[index]
            prefix = encode_integer(index, 7)
            prefix[0] |= 0x80
            return bytes(prefix)

        bits = 6 if not sensitive else 4
        prefix = encode_integer(index, bits)
        prefix[0] |= index_prefix
        encoded = bytes(prefix) + _encode_string(value, huffman)
        if not sensitive:
            self.header_table.add(matched_name, value)
        return encoded


class Decoder:
    def __init__(self, max_header_list_size: int = 65536) -> None:
        self.header_table = HeaderTable()
        self.max_header_list_size = max_header_list_size
        self.max_allowed_table_size = self.header_table.maxsize

    @property
    def header_table_size(self) -> int:
        return self.header_table.maxsize

    @header_table_size.setter
    def header_table_size(self, value: int) -> None:
        self.header_table.maxsize = value

    def _string(self, data: memoryview, offset: int) -> tuple[bytes, int]:
        if offset >= len(data):
            raise HPACKDecodingError("Truncated header block")
        huffman = bool(data[offset] & 0x80)
        length, consumed = decode_integer(data[offset:], 7)
        start = offset + consumed
        end = start + length
        if end > len(data):
            raise HPACKDecodingError("Truncated header block")
        raw = data[start:end].tobytes()
        if huffman:
            try:
                raw = huffman_decode(raw)
            except ValueError as error:
                raise HPACKDecodingError("Invalid Huffman string") from error
        return raw, consumed + length

    def _literal(
        self, data: memoryview, offset: int, should_index: bool,
    ) -> tuple[HeaderTuple, int]:
        first = data[offset]
        prefix_bits = 6 if should_index else 4
        indexed_name = first & ((1 << prefix_bits) - 1)
        never_index = not should_index and bool(first & 0x10)
        consumed = 0
        if indexed_name:
            index, integer_size = decode_integer(data[offset:], prefix_bits)
            name = self.header_table.get_by_index(index)[0]
            consumed += integer_size
        else:
            consumed = 1
            name, string_size = self._string(data, offset + consumed)
            consumed += string_size
        value, string_size = self._string(data, offset + consumed)
        consumed += string_size
        header_type = NeverIndexedHeaderTuple if never_index else HeaderTuple
        header = header_type(name, value)
        if should_index:
            self.header_table.add(name, value)
        return header, consumed

    def decode(self, data: bytes, raw: bool = False):
        view = memoryview(data)
        headers = []
        inflated_size = 0
        offset = 0
        while offset < len(view):
            first = view[offset]
            if first & 0x80:
                index, consumed = decode_integer(view[offset:], 7)
                header = HeaderTuple(*self.header_table.get_by_index(index))
            elif first & 0x40:
                header, consumed = self._literal(view, offset, True)
            elif first & 0x20:
                if headers:
                    raise HPACKDecodingError(
                        "Table size update not at the start of the block"
                    )
                size, consumed = decode_integer(view[offset:], 5)
                if size > self.max_allowed_table_size:
                    raise InvalidTableSizeError(
                        "Encoder exceeded max allowable table size"
                    )
                self.header_table_size = size
                header = None
            else:
                header, consumed = self._literal(view, offset, False)
            offset += consumed
            if header is not None:
                headers.append(header)
                inflated_size += table_entry_size(header[0], header[1])
                if inflated_size > self.max_header_list_size:
                    raise OversizedHeaderListError(
                        f"A header list larger than {self.max_header_list_size} "
                        "has been received"
                    )
        if self.header_table_size > self.max_allowed_table_size:
            raise InvalidTableSizeError(
                "Encoder did not shrink table size to within the max"
            )
        if raw:
            return headers
        try:
            return [
                header.__class__(
                    bytes(header[0]).decode("utf-8"),
                    bytes(header[1]).decode("utf-8"),
                )
                for header in headers
            ]
        except UnicodeDecodeError as error:
            raise HPACKDecodingError("Unable to decode headers as UTF-8") from error
