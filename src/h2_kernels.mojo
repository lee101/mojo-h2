comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def huffman_encoded_size(src: U8Ptr, n: Int, lengths: U8Ptr) -> Int:
    var bits = 0
    for i in range(n):
        bits += Int(lengths[Int(src[i])])
    return (bits + 7) // 8


def huffman_encode(
    src: U8Ptr, n: Int, codes: U32Ptr, lengths: U8Ptr, dst: U8Ptr
) -> Int:
    var bit_buffer = UInt64(0)
    var bit_count = 0
    var dst_index = 0
    for i in range(n):
        var symbol = Int(src[i])
        var width = Int(lengths[symbol])
        bit_buffer = (bit_buffer << UInt64(width)) | UInt64(codes[symbol])
        bit_count += width
        while bit_count >= 8:
            bit_count -= 8
            dst[dst_index] = UInt8((bit_buffer >> UInt64(bit_count)) & 0xFF)
            dst_index += 1
            if bit_count == 0:
                bit_buffer = UInt64(0)
            else:
                bit_buffer &= (UInt64(1) << UInt64(bit_count)) - 1
    if bit_count:
        dst[dst_index] = UInt8(
            (bit_buffer << UInt64(8 - bit_count))
            | ((UInt64(1) << UInt64(8 - bit_count)) - 1)
        )
        dst_index += 1
    return dst_index


def huffman_decode(
    src: U8Ptr,
    n: Int,
    left: I32Ptr,
    right: I32Ptr,
    symbols: I32Ptr,
    accepting: U8Ptr,
    dst: U8Ptr,
    capacity: Int,
) -> Int:
    var node = 0
    var dst_index = 0
    for i in range(n):
        var byte = Int(src[i])
        for shift in range(7, -1, -1):
            if (byte >> shift) & 1:
                node = Int(right[node])
            else:
                node = Int(left[node])
            if node < 0:
                return -1
            var symbol = Int(symbols[node])
            if symbol >= 0:
                if symbol == 256:
                    return -2
                if dst_index >= capacity:
                    return -3
                dst[dst_index] = UInt8(symbol)
                dst_index += 1
                node = 0
    if node != 0 and accepting[node] == 0:
        return -2
    return dst_index


def encode_integer(value: Int, prefix_bits: Int, dst: U8Ptr) -> Int:
    var maximum = (1 << prefix_bits) - 1
    if value < maximum:
        dst[0] = UInt8(value)
        return 1
    dst[0] = UInt8(maximum)
    var remaining = value - maximum
    var index = 1
    while remaining >= 128:
        dst[index] = UInt8((remaining & 127) | 128)
        remaining >>= 7
        index += 1
    dst[index] = UInt8(remaining)
    return index + 1


def decode_integer(src: U8Ptr, n: Int, prefix_bits: Int, value: I64Ptr) -> Int:
    if n == 0:
        return -1
    var maximum = (1 << prefix_bits) - 1
    var number = Int(src[0]) & maximum
    if number < maximum:
        value[0] = Int64(number)
        return 1
    var index = 1
    var shift = 0
    while index < n and index <= 5:
        var byte = Int(src[index])
        number += (byte & 127) << shift
        index += 1
        if byte < 128:
            value[0] = Int64(number)
            return index
        shift += 7
    return -1


def encode_frame_headers(
    lengths: I64Ptr,
    types: U8Ptr,
    flags: U8Ptr,
    stream_ids: I64Ptr,
    dst: U8Ptr,
    n: Int,
):
    for i in range(n):
        var length = Int(lengths[i])
        var stream_id = Int(stream_ids[i]) & 0x7FFFFFFF
        var offset = i * 9
        dst[offset] = UInt8((length >> 16) & 0xFF)
        dst[offset + 1] = UInt8((length >> 8) & 0xFF)
        dst[offset + 2] = UInt8(length & 0xFF)
        dst[offset + 3] = types[i]
        dst[offset + 4] = flags[i]
        dst[offset + 5] = UInt8((stream_id >> 24) & 0x7F)
        dst[offset + 6] = UInt8((stream_id >> 16) & 0xFF)
        dst[offset + 7] = UInt8((stream_id >> 8) & 0xFF)
        dst[offset + 8] = UInt8(stream_id & 0xFF)


def decode_frame_headers(
    src: U8Ptr,
    n: Int,
    lengths: I64Ptr,
    types: U8Ptr,
    flags: U8Ptr,
    stream_ids: I64Ptr,
):
    for i in range(n):
        var offset = i * 9
        lengths[i] = Int64(
            (Int(src[offset]) << 16)
            | (Int(src[offset + 1]) << 8)
            | Int(src[offset + 2])
        )
        types[i] = src[offset + 3]
        flags[i] = src[offset + 4]
        stream_ids[i] = Int64(
            ((Int(src[offset + 5]) & 0x7F) << 24)
            | (Int(src[offset + 6]) << 16)
            | (Int(src[offset + 7]) << 8)
            | Int(src[offset + 8])
        )


@export("mh2_huffman_encoded_size")
def mh2_huffman_encoded_size(src: Int, n: Int, lengths: Int) abi("C") -> Int:
    return huffman_encoded_size(
        U8Ptr(unsafe_from_address=src), n, U8Ptr(unsafe_from_address=lengths)
    )


@export("mh2_huffman_encode")
def mh2_huffman_encode(
    src: Int, n: Int, codes: Int, lengths: Int, dst: Int
) abi("C") -> Int:
    return huffman_encode(
        U8Ptr(unsafe_from_address=src),
        n,
        U32Ptr(unsafe_from_address=codes),
        U8Ptr(unsafe_from_address=lengths),
        U8Ptr(unsafe_from_address=dst),
    )


@export("mh2_huffman_decode")
def mh2_huffman_decode(
    src: Int,
    n: Int,
    left: Int,
    right: Int,
    symbols: Int,
    accepting: Int,
    dst: Int,
    capacity: Int,
) abi("C") -> Int:
    return huffman_decode(
        U8Ptr(unsafe_from_address=src),
        n,
        I32Ptr(unsafe_from_address=left),
        I32Ptr(unsafe_from_address=right),
        I32Ptr(unsafe_from_address=symbols),
        U8Ptr(unsafe_from_address=accepting),
        U8Ptr(unsafe_from_address=dst),
        capacity,
    )


@export("mh2_encode_integer")
def mh2_encode_integer(value: Int, prefix_bits: Int, dst: Int) abi("C") -> Int:
    return encode_integer(value, prefix_bits, U8Ptr(unsafe_from_address=dst))


@export("mh2_decode_integer")
def mh2_decode_integer(
    src: Int, n: Int, prefix_bits: Int, value: Int
) abi("C") -> Int:
    return decode_integer(
        U8Ptr(unsafe_from_address=src),
        n,
        prefix_bits,
        I64Ptr(unsafe_from_address=value),
    )


@export("mh2_encode_frame_headers")
def mh2_encode_frame_headers(
    lengths: Int,
    types: Int,
    flags: Int,
    stream_ids: Int,
    dst: Int,
    n: Int,
) abi("C"):
    encode_frame_headers(
        I64Ptr(unsafe_from_address=lengths),
        U8Ptr(unsafe_from_address=types),
        U8Ptr(unsafe_from_address=flags),
        I64Ptr(unsafe_from_address=stream_ids),
        U8Ptr(unsafe_from_address=dst),
        n,
    )


@export("mh2_decode_frame_headers")
def mh2_decode_frame_headers(
    src: Int,
    n: Int,
    lengths: Int,
    types: Int,
    flags: Int,
    stream_ids: Int,
) abi("C"):
    decode_frame_headers(
        U8Ptr(unsafe_from_address=src),
        n,
        I64Ptr(unsafe_from_address=lengths),
        U8Ptr(unsafe_from_address=types),
        U8Ptr(unsafe_from_address=flags),
        I64Ptr(unsafe_from_address=stream_ids),
    )
