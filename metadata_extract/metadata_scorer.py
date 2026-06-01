"""
metadata_scorer.py

Lightweight metadata analysis for AI image detection. Checks two things only:
    1. Whether EXIF data exists in a JPEG file (absence is a mild AI signal)
    2. Whether the Software tag names a known AI generation tool

Returns a score in [0.0, 1.0]:
    0.0 - looks like a real camera image
    1.0 - strong AI signal in metadata

Because many images (real or AI) have stripped metadata, this module is treated
as a low-weight bonus signal in score fusion, not a primary indicator.
It only makes a strong contribution when an AI software tag is present.
"""

from __future__ import annotations

# EXIF constants
EXIF_APP1_MARKER         = b"\xFF\xE1"   # start of an EXIF APP1 segment in JPEG
EXIF_APP1_HEADER_LEN     = 14            # minimum bytes needed to read endianness and IFD offset
EXIF_IFD_OFFSET_BYTES    = 4             # byte position of the IFD offset within the header
EXIF_LITTLE_ENDIAN_MARKER = b"II"
EXIF_BIG_ENDIAN_MARKER    = b"MM"
EXIF_TAG_SOFTWARE        = 0x0131        # EXIF tag ID for the Software field
EXIF_DTYPE_ASCII         = 2             # EXIF data type: ASCII string
EXIF_ENTRY_SIZE          = 12            # bytes per IFD entry
EXIF_INLINE_MAX_BYTES    = 4             # values <= 4 bytes are stored inline in the entry

# How far into the file to scan for the EXIF APP1 marker
JPEG_HEADER_SCAN_LIMIT = 65536

# Score returned when the image format cannot be checked (PNG, WEBP, etc.)
SCORE_FORMAT_SKIP = 0.5

# Scores for different EXIF findings
SCORE_NO_EXIF     = 0.35   # JPEG with no EXIF at all (mild AI signal)
SCORE_NO_SOFTWARE = 0.20   # EXIF present but no Software tag
SCORE_AI_SOFTWARE = 0.95   # Software tag names a known AI tool
SCORE_REAL_CAMERA = 0.05   # Software tag looks like a real camera or editor

# Known AI generation software keywords (matched against the Software tag, lowercase)
_AI_SOFTWARE_KEYWORDS = [
    "stablediffusion", "stable diffusion",
    "midjourney",
    "dall-e", "dalle",
    "firefly",
    "novelai",
    "comfyui",
    "automatic1111",
    "invokeai",
    "diffusion",
    "generative",
]


def _detect_format(data: bytes) -> str:
    """Returns the image format string based on magic bytes."""
    if data.startswith(b"\xFF\xD8"):
        return "JPEG"
    if data.startswith(b"\x89PNG"):
        return "PNG"
    if data.startswith(b"RIFF") and b"WEBP" in data[8:16]:
        return "WEBP"
    return "UNKNOWN"


def _read_software_tag(data: bytes) -> str | None:
    """
    Locates the EXIF APP1 segment in JPEG bytes and reads the Software tag (0x0131).
    Returns the software string if found, or None otherwise.

    bytes

    0-1         Tag ID
    2-3         Data type
    4-7         Count
    8-11        Value or pointer to value
    """
    exif_pos = data.find(EXIF_APP1_MARKER)
    if exif_pos == -1:
        return None

    # Read segment length and extract that many bytes
    seg_len   = int.from_bytes(data[exif_pos + 2: exif_pos + 4], "big")
    exif_data = data[exif_pos + 4: exif_pos + 2 + seg_len]

    if len(exif_data) < EXIF_APP1_HEADER_LEN:
        return None

    # Determine byte order from the endianness marker
    endian_marker = exif_data[6:8]
    # An endian is the byte order: little first byte = msb
    #                               big last byte = msb
    if endian_marker == EXIF_LITTLE_ENDIAN_MARKER:
        endian = "little"
    elif endian_marker == EXIF_BIG_ENDIAN_MARKER:
        endian = "big"
    else:
        return None

    # Jump to the IFD (Image File Directory) and read its entries
    ifd_offset = int.from_bytes(exif_data[10:14], endian)
    ifd0_pos   = 6 + ifd_offset
    if ifd0_pos + 2 > len(exif_data):
        return None

    # The IFD starts with a 2-byte entry count; each entry is EXIF_ENTRY_SIZE bytes
    num_entries   = int.from_bytes(exif_data[ifd0_pos: ifd0_pos + 2], endian)
    entries_start = ifd0_pos + 2

    for i in range(num_entries):
        offset = entries_start + i * EXIF_ENTRY_SIZE
        if offset + EXIF_ENTRY_SIZE > len(exif_data):
            break

        entry  = exif_data[offset: offset + EXIF_ENTRY_SIZE]
        tag_id = int.from_bytes(entry[0:2], endian)
        if tag_id != EXIF_TAG_SOFTWARE:
            continue

        dtype = int.from_bytes(entry[2:4], endian)
        if dtype != EXIF_DTYPE_ASCII:
            continue

        # For values longer than EXIF_INLINE_MAX_BYTES, the entry holds an offset to the data
        count  = int.from_bytes(entry[4:8], endian)
        value_raw = entry[8:12]
        if count <= EXIF_INLINE_MAX_BYTES:
            raw = value_raw[:count]
        else:
            data_offset = int.from_bytes(value_raw, endian)
            data_pos    = 6 + data_offset
            raw         = exif_data[data_pos: data_pos + count]

        # ASCII strings are null-terminated; split to get the actual text
        return raw.split(b"\x00")[0].decode("ascii", errors="ignore").strip()

    return None


def analyze_metadata(image_path: str) -> dict:
    """
    Analyzes image metadata and returns a score reflecting the likelihood of AI generation.
    Higher score = more likely AI-generated.
    """
    try:
        with open(image_path, "rb") as f:
            data = f.read()
    except Exception as e:
        return {"score": SCORE_FORMAT_SKIP, "reason": f"Could not read file: {e}"}

    fmt = _detect_format(data)

    # PNG and WEBP do not embed EXIF in the same way as JPEG - skip EXIF checks
    if fmt != "JPEG":
        return {"score": SCORE_FORMAT_SKIP, "reason": f"{fmt} format - metadata check skipped"}

    # Check for the presence of an EXIF segment (scan only the file header area)
    has_exif = EXIF_APP1_MARKER in data[:JPEG_HEADER_SCAN_LIMIT]
    if not has_exif:
        return {
            "score":  SCORE_NO_EXIF,
            "reason": "JPEG with no EXIF data (mild AI signal)",
        }

    software = _read_software_tag(data)
    if software is None:
        return {
            "score":  SCORE_NO_SOFTWARE,
            "reason": "EXIF present but no Software tag found",
        }

    software_lower = software.lower()
    ai_hit = next(
        (kw for kw in _AI_SOFTWARE_KEYWORDS if kw in software_lower), None
    )

    if ai_hit:
        return {
            "score":  SCORE_AI_SOFTWARE,
            "reason": f"Software tag identifies AI tool: '{software}'",
        }

    return {
        "score":  SCORE_REAL_CAMERA,
        "reason": f"Software tag looks like a real tool: '{software}'",
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    default_path = (
        Path(__file__).resolve().parent.parent / "images" / "testImages" / "aiImageTest.jpg"
    )
    path   = sys.argv[1] if len(sys.argv) > 1 else str(default_path)
    result = analyze_metadata(path)
    print(f"Score  : {result['score']}")
    print(f"Reason : {result['reason']}")
