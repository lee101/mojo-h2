class HPACKError(Exception):
    pass


class HPACKDecodingError(HPACKError):
    pass


class InvalidTableIndexError(HPACKDecodingError):
    pass


class InvalidTableIndex(InvalidTableIndexError):
    pass


class OversizedHeaderListError(HPACKDecodingError):
    pass


class InvalidTableSizeError(HPACKDecodingError):
    pass


class HyperframeError(Exception):
    pass


class UnknownFrameError(HyperframeError):
    def __init__(self, frame_type: int, length: int) -> None:
        self.frame_type = frame_type
        self.length = length

    def __str__(self) -> str:
        return (
            f"UnknownFrameError: Unknown frame type 0x{self.frame_type:X} "
            f"received, length {self.length} bytes"
        )


class InvalidPaddingError(HyperframeError):
    pass


class InvalidFrameError(HyperframeError):
    pass


class InvalidDataError(HyperframeError):
    pass
