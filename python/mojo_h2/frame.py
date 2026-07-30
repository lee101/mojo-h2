"""HTTP/2 frame objects with hyperframe-compatible names and signatures."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator, MutableSet
from typing import Any, NamedTuple

from .exceptions import (
    InvalidDataError,
    InvalidFrameError,
    InvalidPaddingError,
    UnknownFrameError,
)

FRAME_MAX_LEN = 2**14
FRAME_MAX_ALLOWED_LEN = 2**24 - 1
_FRAME_HEADER = struct.Struct("!HBBBL")


class Flag(NamedTuple):
    name: str
    bit: int


class Flags(MutableSet):
    def __init__(
        self, defined_flags: Iterable[Flag] | frozenset[str]
    ) -> None:
        if isinstance(defined_flags, frozenset):
            self._valid_flags = defined_flags
        else:
            self._valid_flags = frozenset(flag.name for flag in defined_flags)
        self._flags: set[str] = set()

    def __contains__(self, value: object) -> bool:
        return value in self._flags

    def __iter__(self) -> Iterator[str]:
        return iter(self._flags)

    def __len__(self) -> int:
        return len(self._flags)

    def __repr__(self) -> str:
        return repr(sorted(self._flags))

    def add(self, value: str) -> None:
        if value not in self._valid_flags:
            raise ValueError(
                f"Unexpected flag: {value}. Valid flags are: {self._valid_flags}"
            )
        self._flags.add(value)

    def discard(self, value: str) -> None:
        self._flags.discard(value)


class Frame:
    defined_flags: list[Flag] = []
    _valid_flags: frozenset[str] = frozenset()
    type: int | None = None
    stream_association = "either"

    def __init_subclass__(cls) -> None:
        cls._valid_flags = frozenset(flag.name for flag in cls.defined_flags)

    def __init__(self, stream_id: int, flags: Iterable[str] = ()) -> None:
        self.stream_id = stream_id
        self.flags = Flags(self._valid_flags)
        self.body_len = 0
        for flag in flags:
            self.flags.add(flag)
        if not stream_id and self.stream_association == "has-stream":
            raise InvalidDataError(
                f"Stream ID must be non-zero for {type(self).__name__}"
            )
        if stream_id and self.stream_association == "no-stream":
            raise InvalidDataError(
                f"Stream ID must be zero for {type(self).__name__} "
                f"with stream_id={stream_id}"
            )

    @staticmethod
    def parse_frame_header(
        header: memoryview, strict: bool = False,
    ) -> tuple["Frame", int]:
        if len(header) != 9:
            raise InvalidFrameError("Invalid frame header")
        length_high, length_low, frame_type, flag_byte, stream_id = (
            _FRAME_HEADER.unpack(header)
        )
        length = (length_high << 8) | length_low
        try:
            frame = FRAMES[frame_type](stream_id & 0x7FFFFFFF)
        except KeyError as error:
            if strict:
                raise UnknownFrameError(frame_type, length) from error
            frame = ExtensionFrame(
                type=frame_type, stream_id=stream_id & 0x7FFFFFFF
            )
        frame.parse_flags(flag_byte)
        return frame, length

    @staticmethod
    def explain(data: memoryview) -> tuple["Frame", int]:
        frame, length = Frame.parse_frame_header(data[:9])
        frame.parse_body(data[9:9 + length])
        print(frame)
        return frame, length

    def parse_flags(self, flag_byte: int) -> Flags:
        for flag in self.defined_flags:
            if flag_byte & flag.bit:
                self.flags.add(flag.name)
        return self.flags

    def serialize(self) -> bytes:
        body = self.serialize_body()
        self.body_len = len(body)
        flag_byte = 0
        if self.flags:
            flag_byte = sum(
                flag.bit for flag in self.defined_flags
                if flag.name in self.flags
            )
        header = _FRAME_HEADER.pack(
            (self.body_len >> 8) & 0xFFFF,
            self.body_len & 0xFF,
            self.type,
            flag_byte,
            self.stream_id & 0x7FFFFFFF,
        )
        return header + body

    def serialize_body(self) -> bytes:
        raise NotImplementedError

    def parse_body(self, data: memoryview) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(stream_id={self.stream_id}, "
            f"flags={self.flags!r})"
        )


def _kwargs(kwargs: dict[str, Any]) -> tuple[Iterable[str], int, int, int, bool]:
    return (
        kwargs.pop("flags", ()),
        kwargs.pop("pad_length", 0),
        kwargs.pop("depends_on", 0),
        kwargs.pop("stream_weight", 0),
        kwargs.pop("exclusive", False),
    )


def _padding(frame, data: memoryview) -> int:
    if "PADDED" not in frame.flags:
        return 0
    if not data:
        raise InvalidFrameError("Invalid Padding data")
    frame.pad_length = data[0]
    return 1


def _priority(frame, data: memoryview) -> int:
    if len(data) < 5:
        raise InvalidFrameError("Invalid Priority data")
    dependency, frame.stream_weight = struct.unpack("!LB", data[:5])
    frame.exclusive = bool(dependency >> 31)
    frame.depends_on = dependency & 0x7FFFFFFF
    return 5


def _priority_bytes(frame) -> bytes:
    dependency = frame.depends_on | (0x80000000 if frame.exclusive else 0)
    return struct.pack("!LB", dependency, frame.stream_weight)


class DataFrame(Frame):
    defined_flags = [Flag("END_STREAM", 0x01), Flag("PADDED", 0x08)]
    type = 0x00
    stream_association = "has-stream"

    def __init__(self, stream_id: int, data: bytes = b"", **kwargs: Any) -> None:
        flags = kwargs.pop("flags", ())
        self.pad_length = kwargs.pop("pad_length", 0)
        super().__init__(stream_id, flags)
        self.data = data

    def serialize_body(self) -> bytes:
        data = self.data.tobytes() if isinstance(self.data, memoryview) else self.data
        if "PADDED" not in self.flags:
            return data
        return bytes([self.pad_length]) + data + b"\0" * self.pad_length

    def parse_body(self, data: memoryview) -> None:
        offset = _padding(self, data)
        self.body_len = len(data)
        if self.pad_length and self.pad_length >= self.body_len:
            raise InvalidPaddingError("Padding is too long.")
        self.data = data[offset:len(data) - self.pad_length].tobytes()

    @property
    def flow_controlled_length(self) -> int:
        padding = self.pad_length + 1 if "PADDED" in self.flags else 0
        return len(self.data) + padding


class HeadersFrame(Frame):
    defined_flags = [
        Flag("END_STREAM", 0x01), Flag("END_HEADERS", 0x04),
        Flag("PADDED", 0x08), Flag("PRIORITY", 0x20),
    ]
    type = 0x01
    stream_association = "has-stream"

    def __init__(self, stream_id: int, data: bytes = b"", **kwargs: Any) -> None:
        (
            flags, self.pad_length, self.depends_on,
            self.stream_weight, self.exclusive,
        ) = _kwargs(kwargs)
        super().__init__(stream_id, flags)
        self.data = data

    def serialize_body(self) -> bytes:
        pieces = [bytes([self.pad_length])] if "PADDED" in self.flags else []
        if "PRIORITY" in self.flags:
            pieces.append(_priority_bytes(self))
        pieces.extend([self.data, b"\0" * self.pad_length])
        return b"".join(pieces)

    def parse_body(self, data: memoryview) -> None:
        offset = _padding(self, data)
        if "PRIORITY" in self.flags:
            offset += _priority(self, data[offset:])
        self.body_len = len(data) - (1 if "PADDED" in self.flags else 0)
        if self.pad_length and self.pad_length >= self.body_len:
            raise InvalidPaddingError("Padding is too long.")
        self.data = data[offset:len(data) - self.pad_length].tobytes()


class PriorityFrame(Frame):
    type = 0x02
    stream_association = "has-stream"

    def __init__(
        self, stream_id: int, depends_on: int = 0, stream_weight: int = 0,
        exclusive: bool = False, **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.depends_on = depends_on
        self.stream_weight = stream_weight
        self.exclusive = exclusive

    def serialize_body(self) -> bytes:
        return _priority_bytes(self)

    def parse_body(self, data: memoryview) -> None:
        if len(data) > 5:
            raise InvalidFrameError(
                f"PRIORITY must have 5 byte body: actual length {len(data)}."
            )
        _priority(self, data)
        self.body_len = 5


class RstStreamFrame(Frame):
    type = 0x03
    stream_association = "has-stream"

    def __init__(
        self, stream_id: int, error_code: int = 0, **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.error_code = error_code

    def serialize_body(self) -> bytes:
        return struct.pack("!L", self.error_code)

    def parse_body(self, data: memoryview) -> None:
        if len(data) != 4:
            raise InvalidFrameError(
                f"RST_STREAM must have 4 byte body: actual length {len(data)}."
            )
        self.error_code = struct.unpack("!L", data)[0]
        self.body_len = 4


class SettingsFrame(Frame):
    defined_flags = [Flag("ACK", 0x01)]
    type = 0x04
    stream_association = "no-stream"
    HEADER_TABLE_SIZE = 0x01
    ENABLE_PUSH = 0x02
    MAX_CONCURRENT_STREAMS = 0x03
    INITIAL_WINDOW_SIZE = 0x04
    MAX_FRAME_SIZE = 0x05
    MAX_HEADER_LIST_SIZE = 0x06
    ENABLE_CONNECT_PROTOCOL = 0x08

    def __init__(
        self, stream_id: int = 0, settings: dict[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        flags = kwargs.pop("flags", ())
        if settings and "ACK" in flags:
            raise InvalidDataError("Settings must be empty if ACK flag is set.")
        super().__init__(stream_id, flags)
        self.settings = settings or {}

    def serialize_body(self) -> bytes:
        return b"".join(
            struct.pack("!HL", setting & 0xFF, value)
            for setting, value in self.settings.items()
        )

    def parse_body(self, data: memoryview) -> None:
        if "ACK" in self.flags and data:
            raise InvalidDataError(
                f"SETTINGS ack frame must not have payload: got {len(data)} bytes"
            )
        if len(data) % 6:
            raise InvalidFrameError("Invalid SETTINGS body")
        for offset in range(0, len(data), 6):
            setting, value = struct.unpack("!HL", data[offset:offset + 6])
            self.settings[setting] = value
        self.body_len = len(data)


class PushPromiseFrame(Frame):
    defined_flags = [Flag("END_HEADERS", 0x04), Flag("PADDED", 0x08)]
    type = 0x05
    stream_association = "has-stream"

    def __init__(
        self, stream_id: int, promised_stream_id: int = 0, data: bytes = b"",
        **kwargs: Any,
    ) -> None:
        flags, self.pad_length, _, _, _ = _kwargs(kwargs)
        super().__init__(stream_id, flags)
        self.promised_stream_id = promised_stream_id
        self.data = data

    def serialize_body(self) -> bytes:
        prefix = bytes([self.pad_length]) if "PADDED" in self.flags else b""
        return (
            prefix + struct.pack("!L", self.promised_stream_id)
            + self.data + b"\0" * self.pad_length
        )

    def parse_body(self, data: memoryview) -> None:
        offset = _padding(self, data)
        if len(data) < offset + 4:
            raise InvalidFrameError("Invalid PUSH_PROMISE body")
        self.promised_stream_id = struct.unpack("!L", data[offset:offset + 4])[0]
        self.data = data[offset + 4:len(data) - self.pad_length].tobytes()
        self.body_len = len(data)
        if self.promised_stream_id == 0 or self.promised_stream_id % 2:
            raise InvalidDataError(
                f"Invalid PUSH_PROMISE promised stream id: "
                f"{self.promised_stream_id}"
            )
        if self.pad_length and self.pad_length >= self.body_len:
            raise InvalidPaddingError("Padding is too long.")


class PingFrame(Frame):
    defined_flags = [Flag("ACK", 0x01)]
    type = 0x06
    stream_association = "no-stream"

    def __init__(
        self, stream_id: int = 0, opaque_data: bytes = b"", **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.opaque_data = opaque_data

    def serialize_body(self) -> bytes:
        if len(self.opaque_data) > 8:
            raise InvalidFrameError(
                f"PING frame may not have more than 8 bytes of data, "
                f"got {len(self.opaque_data)}"
            )
        return self.opaque_data + b"\0" * (8 - len(self.opaque_data))

    def parse_body(self, data: memoryview) -> None:
        if len(data) != 8:
            raise InvalidFrameError(
                f"PING frame must have 8 byte length: got {len(data)}"
            )
        self.opaque_data = data.tobytes()
        self.body_len = 8


class GoAwayFrame(Frame):
    type = 0x07
    stream_association = "no-stream"

    def __init__(
        self, stream_id: int = 0, last_stream_id: int = 0, error_code: int = 0,
        additional_data: bytes = b"", **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.last_stream_id = last_stream_id
        self.error_code = error_code
        self.additional_data = additional_data

    def serialize_body(self) -> bytes:
        return struct.pack(
            "!LL", self.last_stream_id & 0x7FFFFFFF, self.error_code
        ) + self.additional_data

    def parse_body(self, data: memoryview) -> None:
        if len(data) < 8:
            raise InvalidFrameError("Invalid GOAWAY body.")
        self.last_stream_id, self.error_code = struct.unpack("!LL", data[:8])
        self.additional_data = data[8:].tobytes()
        self.body_len = len(data)


class WindowUpdateFrame(Frame):
    type = 0x08

    def __init__(
        self, stream_id: int, window_increment: int = 0, **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.window_increment = window_increment

    def serialize_body(self) -> bytes:
        return struct.pack("!L", self.window_increment & 0x7FFFFFFF)

    def parse_body(self, data: memoryview) -> None:
        if len(data) > 4:
            raise InvalidFrameError(
                f"WINDOW_UPDATE frame must have 4 byte length: got {len(data)}"
            )
        try:
            self.window_increment = struct.unpack("!L", data)[0]
        except struct.error as error:
            raise InvalidFrameError("Invalid WINDOW_UPDATE body") from error
        if not 1 <= self.window_increment <= 2**31 - 1:
            raise InvalidDataError(
                "WINDOW_UPDATE increment must be between 1 to 2^31-1"
            )
        self.body_len = 4


class ContinuationFrame(Frame):
    defined_flags = [Flag("END_HEADERS", 0x04)]
    type = 0x09
    stream_association = "has-stream"

    def __init__(self, stream_id: int, data: bytes = b"", **kwargs: Any) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.data = data

    def serialize_body(self) -> bytes:
        return self.data

    def parse_body(self, data: memoryview) -> None:
        self.data = data.tobytes()
        self.body_len = len(data)


class AltSvcFrame(Frame):
    type = 0x0A

    def __init__(
        self, stream_id: int, origin: bytes = b"", field: bytes = b"",
        **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        if not isinstance(origin, bytes):
            raise InvalidDataError("AltSvc origin must be a bytestring.")
        if not isinstance(field, bytes):
            raise InvalidDataError("AltSvc field must be a bytestring.")
        self.origin = origin
        self.field = field

    def serialize_body(self) -> bytes:
        return struct.pack("!H", len(self.origin)) + self.origin + self.field

    def parse_body(self, data: memoryview) -> None:
        try:
            origin_length = struct.unpack("!H", data[:2])[0]
        except struct.error as error:
            raise InvalidFrameError("Invalid ALTSVC frame body.") from error
        self.origin = data[2:2 + origin_length].tobytes()
        if len(self.origin) != origin_length:
            raise InvalidFrameError("Invalid ALTSVC frame body.")
        self.field = data[2 + origin_length:].tobytes()
        self.body_len = len(data)


class ExtensionFrame(Frame):
    def __init__(
        self, type: int, stream_id: int, flag_byte: int = 0,
        body: bytes = b"", **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, kwargs.pop("flags", ()))
        self.type = type
        self.flag_byte = flag_byte
        self.body = body

    def parse_flags(self, flag_byte: int) -> None:
        self.flag_byte = flag_byte

    def serialize_body(self) -> bytes:
        return self.body

    def parse_body(self, data: memoryview) -> None:
        self.body = data.tobytes()
        self.body_len = len(data)

    def serialize(self) -> bytes:
        return _FRAME_HEADER.pack(
            (self.body_len >> 8) & 0xFFFF,
            self.body_len & 0xFF,
            self.type,
            self.flag_byte,
            self.stream_id & 0x7FFFFFFF,
        ) + self.body


_FRAME_CLASSES = [
    DataFrame, HeadersFrame, PriorityFrame, RstStreamFrame, SettingsFrame,
    PushPromiseFrame, PingFrame, GoAwayFrame, WindowUpdateFrame,
    ContinuationFrame, AltSvcFrame,
]
FRAMES = {frame.type: frame for frame in _FRAME_CLASSES}
