from __future__ import annotations

import struct

import pytest
from h2.config import H2Configuration
from h2.connection import H2Connection
from hyperframe import frame as reference

from mojo_h2 import frame
from mojo_h2.exceptions import (
    InvalidDataError,
    InvalidFrameError,
    InvalidPaddingError,
    UnknownFrameError,
)
from mojo_h2.hpack import Decoder


def frame_pairs():
    return [
        (frame.DataFrame(1, b"abc", flags=["END_STREAM"]),
         reference.DataFrame(1, b"abc", flags=["END_STREAM"])),
        (frame.DataFrame(1, b"abc", flags=["PADDED"], pad_length=3),
         reference.DataFrame(1, b"abc", flags=["PADDED"], pad_length=3)),
        (frame.HeadersFrame(
            3, b"block", flags=["END_HEADERS", "PRIORITY", "PADDED"],
            pad_length=2, depends_on=1, stream_weight=20, exclusive=True,
         ), reference.HeadersFrame(
            3, b"block", flags=["END_HEADERS", "PRIORITY", "PADDED"],
            pad_length=2, depends_on=1, stream_weight=20, exclusive=True,
         )),
        (frame.PriorityFrame(3, 1, 42, True),
         reference.PriorityFrame(3, 1, 42, True)),
        (frame.RstStreamFrame(1, 7), reference.RstStreamFrame(1, 7)),
        (frame.SettingsFrame(settings={1: 4096, 4: 65535}),
         reference.SettingsFrame(settings={1: 4096, 4: 65535})),
        (frame.SettingsFrame(flags=["ACK"]),
         reference.SettingsFrame(flags=["ACK"])),
        (frame.PushPromiseFrame(
            1, 2, b"head", flags=["PADDED", "END_HEADERS"], pad_length=2,
         ), reference.PushPromiseFrame(
            1, 2, b"head", flags=["PADDED", "END_HEADERS"], pad_length=2,
         )),
        (frame.PingFrame(opaque_data=b"abcdefgh", flags=["ACK"]),
         reference.PingFrame(opaque_data=b"abcdefgh", flags=["ACK"])),
        (frame.GoAwayFrame(
            last_stream_id=5, error_code=2, additional_data=b"bye",
         ), reference.GoAwayFrame(
            last_stream_id=5, error_code=2, additional_data=b"bye",
         )),
        (frame.WindowUpdateFrame(0, 12345),
         reference.WindowUpdateFrame(0, 12345)),
        (frame.ContinuationFrame(1, b"zzz", flags=["END_HEADERS"]),
         reference.ContinuationFrame(1, b"zzz", flags=["END_HEADERS"])),
        (frame.AltSvcFrame(0, b"https://x", b'h2=":443"'),
         reference.AltSvcFrame(0, b"https://x", b'h2=":443"')),
        (frame.ExtensionFrame(99, 3, 7, b"body"),
         reference.ExtensionFrame(99, 3, 7, b"body")),
    ]


@pytest.mark.parametrize(("ours", "upstream"), frame_pairs())
def test_frame_serialization_is_byte_identical(ours, upstream):
    assert ours.serialize() == upstream.serialize()


@pytest.mark.parametrize(("ours", "upstream"), frame_pairs()[:-1])
def test_frame_parse_roundtrip_matches_upstream(ours, upstream):
    wire = upstream.serialize()
    parsed, length = frame.Frame.parse_frame_header(memoryview(wire[:9]))
    reference_parsed, reference_length = reference.Frame.parse_frame_header(
        memoryview(wire[:9])
    )
    assert length == reference_length == len(wire) - 9
    parsed.parse_body(memoryview(wire[9:]))
    reference_parsed.parse_body(memoryview(wire[9:]))
    assert parsed.serialize() == reference_parsed.serialize() == wire


def test_unknown_frame_strict_and_non_strict():
    wire = bytes.fromhex("000004630700000003") + b"body"
    parsed, length = frame.Frame.parse_frame_header(memoryview(wire[:9]))
    assert isinstance(parsed, frame.ExtensionFrame)
    assert length == 4
    parsed.parse_body(memoryview(wire[9:]))
    assert parsed.serialize() == wire
    with pytest.raises(UnknownFrameError):
        frame.Frame.parse_frame_header(memoryview(wire[:9]), strict=True)


def test_frame_validation_errors():
    with pytest.raises(InvalidDataError):
        frame.DataFrame(0)
    with pytest.raises(InvalidDataError):
        frame.PingFrame(1)
    with pytest.raises(InvalidFrameError):
        frame.PingFrame(opaque_data=b"123456789").serialize()
    with pytest.raises(InvalidPaddingError):
        parsed = frame.DataFrame(1, flags=["PADDED"])
        parsed.parse_body(memoryview(b"\x04abc"))
    with pytest.raises(ValueError):
        frame.DataFrame(1, flags=["NOT_A_FLAG"])


def test_data_flow_controlled_length():
    plain = frame.DataFrame(1, b"abc")
    padded = frame.DataFrame(1, b"abc", flags=["PADDED"], pad_length=5)
    assert plain.flow_controlled_length == 3
    assert padded.flow_controlled_length == 9


def test_plain_data_memoryview_serialization_updates_body_length():
    payload = memoryview(b"abcdef")[1:5]
    ours = frame.DataFrame(3, payload)
    upstream = reference.DataFrame(3, payload)
    assert ours.serialize() == upstream.serialize()
    assert ours.body_len == upstream.body_len == len(payload)


@pytest.mark.parametrize("size", [0, 1, 255, 256, 65535, 65536])
def test_scalar_data_header_length_boundaries(size):
    payload = b"x" * size
    wire = frame.DataFrame(0xFFFFFFFF, payload).serialize()
    assert wire[:9] == struct.pack(
        "!HBBBL", (size >> 8) & 0xFFFF, size & 0xFF, 0, 0, 0x7FFFFFFF
    )
    assert wire[9:] == payload


def test_upstream_h2_client_preface_and_settings_parse():
    connection = H2Connection(
        config=H2Configuration(client_side=True, header_encoding=None)
    )
    connection.initiate_connection()
    wire = connection.data_to_send()
    assert wire.startswith(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
    settings_wire = wire[24:]
    parsed, length = frame.Frame.parse_frame_header(memoryview(settings_wire[:9]))
    assert isinstance(parsed, frame.SettingsFrame)
    parsed.parse_body(memoryview(settings_wire[9:9 + length]))
    upstream, upstream_length = reference.Frame.parse_frame_header(
        memoryview(settings_wire[:9])
    )
    upstream.parse_body(memoryview(settings_wire[9:9 + upstream_length]))
    assert parsed.settings == upstream.settings


def test_upstream_h2_headers_decode_with_mojo_h2():
    connection = H2Connection(
        config=H2Configuration(client_side=True, header_encoding=None)
    )
    connection.initiate_connection()
    connection.data_to_send()
    headers = [
        (b":method", b"GET"), (b":scheme", b"https"), (b":authority", b"example.com"),
        (b":path", b"/resource"), (b"user-agent", b"mojo-h2-test"),
    ]
    connection.send_headers(1, headers, end_stream=True)
    wire = connection.data_to_send()
    parsed, length = frame.Frame.parse_frame_header(memoryview(wire[:9]))
    assert isinstance(parsed, frame.HeadersFrame)
    parsed.parse_body(memoryview(wire[9:9 + length]))
    assert Decoder().decode(parsed.data, raw=True) == headers
