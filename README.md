# mojo-h2

HTTP/2 wire framing and HPACK header compression backed by
[Mojo](https://www.modular.com/mojo), callable from Python.

Python's `h2` package deliberately delegates these layers to `hyperframe` and
`hpack`. This repo follows that same boundary: its frame classes mirror
`hyperframe.frame`, its stateful `Encoder` and `Decoder` mirror `hpack`, and the
compute-heavy Huffman, integer, and batched common-frame-header codecs run in a
single Mojo shared library.

```python
from mojo_h2 import Decoder, Encoder
from mojo_h2.frame import HeadersFrame

headers = [
    (":method", "GET"),
    (":scheme", "https"),
    (":authority", "example.com"),
    (":path", "/"),
]

block = Encoder().encode(headers)
wire = HeadersFrame(1, data=block, flags=["END_HEADERS", "END_STREAM"]).serialize()

parsed, length = HeadersFrame.parse_frame_header(memoryview(wire[:9]))
parsed.parse_body(memoryview(wire[9:9 + length]))
assert Decoder().decode(parsed.data) == headers
```

## Coverage

The HPACK implementation covers:

- RFC 7541 integer encoding and decoding;
- the fixed RFC Huffman code, including EOS and padding validation;
- indexed, incremental-indexed, non-indexed, and never-indexed fields;
- the 61-entry static table and stateful dynamic table with eviction;
- dynamic table-size updates and decoder size limits;
- `Encoder`, `Decoder`, `HeaderTuple`, and `NeverIndexedHeaderTuple` names and
  call signatures compatible with `hpack`.

The framing implementation covers the nine-byte common header and all frame
classes implemented by `hyperframe` 6.1:

| wire type | class |
| --- | --- |
| DATA | `DataFrame` |
| HEADERS | `HeadersFrame` |
| PRIORITY | `PriorityFrame` |
| RST_STREAM | `RstStreamFrame` |
| SETTINGS | `SettingsFrame` |
| PUSH_PROMISE | `PushPromiseFrame` |
| PING | `PingFrame` |
| GOAWAY | `GoAwayFrame` |
| WINDOW_UPDATE | `WindowUpdateFrame` |
| CONTINUATION | `ContinuationFrame` |
| ALTSVC | `AltSvcFrame` |
| unknown extension types | `ExtensionFrame` |

`encode_frame_headers` and `decode_frame_headers` operate on whole batches,
which is useful for trace analyzers, proxies, and protocol tooling.

Not covered are `h2.connection.H2Connection`, stream lifecycle, flow-control
state, event dispatch, configuration, and connection error handling. This is
not a replacement for the full `h2` state machine. It is a compatible
replacement for the framing and HPACK layers in the stated subset. HTTP/3
QPACK is also out of scope.

## Install and test

The repository pins its Mojo nightly and carries all test dependencies:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` emits `dist/libmojo-h2.so`. Imports rebuild it only when the
source is newer. A deployed copy can set `MOJO_H2_LIB` to an already-built
shared library.

The test suite checks published RFC 7541 examples, randomized stateful HPACK
sequences, invalid encodings, every frame type byte-for-byte
against `hpack` 4.2 and `hyperframe` 6.1, and frames emitted by a real
`h2.H2Connection` 4.4.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14. These are best-of-three wall-clock
times from the complete locked run.

| case | mojo-h2 | upstream | upstream / Mojo |
| --- | ---: | ---: | ---: |
| Huffman encode, 128 KiB | 0.46 ms | 2534.72 ms | 5471.58x faster |
| Huffman decode, 106 KiB | 2.48 ms | 35.74 ms | 14.43x faster |
| HPACK encode, one 128 KiB value | 1.01 ms | 2731.07 ms | 2710.57x faster |
| HPACK encode, 10k indexed blocks | 30.78 ms | 92.07 ms | 2.99x faster |
| Frame-header encode, 250k | 1.69 ms | 185.68 ms | 109.99x faster |
| Frame-header decode, 250k | 0.94 ms | 570.56 ms | 608.41x faster |
| DATA serialize, 10k scalar | 13.31 ms | 25.70 ms | 1.93x faster |

The very large encode ratios are real but specific. Upstream `hpack` builds a
literal's complete Huffman bitstream as one ever-growing Python integer, so a
large single value scales poorly. Mojo keeps a bounded 64-bit bit accumulator.
Normal headers are much smaller, so the absolute saving per value is smaller.

An indexed HPACK field is only one or a few bytes, and a scalar DATA header is
nine bytes. These paths use direct table lookups and scalar packing so they do
not pay for NumPy setup or a ctypes crossing. Static indexed fields bypass the
generic dynamic-table search, while unflagged DATA frames bypass generic flag
serialization. The batch frame-header API still uses Mojo, where that fixed
cost is amortized across the input.

No GPU path is included. Frame packing and Huffman sizing are bandwidth-bound,
Huffman bit packing is serial within a string, and Huffman tree traversal is
branchy with low arithmetic intensity. None offers both roughly two
operations per byte and enough independent work to justify device transfers.
The deficient benchmark paths were also too small or stateful to benefit from
SIMD or thread launch overhead.

## How it works

`src/h2_kernels.mojo` is one compilation unit. Its exported functions use
`@export("name")` with `abi("C")`; Python loads the resulting shared object
with `ctypes`. Buffers cross the ABI as 64-bit integer addresses and are rebuilt
inside Mojo as `UnsafePointer[..., AnyOrigin[mut=True]]`.

Python owns every allocation. Byte inputs are zero-copy NumPy views, output
arrays are allocated once at their exact or conservative maximum size, and
Mojo never retains a pointer after a call. The Huffman decoder traverses a
compact pair of `int32` child arrays plus a symbol array. Frame batches use
structure-of-arrays inputs (`int64` lengths and stream IDs, `uint8` types and
flags) and write packed nine-byte headers contiguously.

HPACK's dynamic table and frame-specific bodies stay in Python because they are
stateful orchestration and small control structures, not compute kernels.
Literal bit packing/unpacking and batched wire-header work stay in Mojo, where
the FFI cost is amortized.

## License

MIT
