"""
metadata_scorer

Lightweight metadata analysis. Checks two things only:
  1. Whether EXIF data exists in the file (JPEG only absence is suspicious)
  2. Whether the Software tag names a known AI generation tool

Returns a score in [0, 1]:
  0.0 - looks like a real camera image
  1.0 - strong AI signal in metadata

Because most images (real or AI) have stripped metadata, this module
is treated as a low-weight bonus signal in score fusion, not a primary
indicator. It only makes a strong contribution when an AI software tag
is actually present.

"""

from __future__ import annotations

# Software strings that are strong indicators of AI generation
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


# Returns image type
def _detect_format(data: bytes) -> str:
    if data.startswith(b"\xFF\xD8"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"RIFF") and b"WEBP" in data[8:16]:
        return "WEBP"
    return "UNKNOWN"

# Digs into the raw byte data to find metaData tags
def _read_software_tag(data: bytes) -> str | None:
    """
    Locates the EXIF APP1 segment in JPEG bytes and reads the
    Software tag
    APP1 is the starting tag of the exif
    """
    exif_pos = data.find(b"\xFF\xE1")
    if exif_pos == -1:
        return None

    # Reads segment length and extracts that many bytes
    seg_len   = int.from_bytes(data[exif_pos + 2: exif_pos + 4], "big")
    exif_data = data[exif_pos + 4: exif_pos + 2 + seg_len]

    if len(exif_data) < 14:
        return None

    # Separates into two exif groups: little endian (ordered by least significant byte)
    # or big endian (ordered by most significant byte)
    endian_marker = exif_data[6:8]
    if endian_marker == b"II":
        endian = "little"
    elif endian_marker == b"MM":
        endian = "big"
    else:
        return None

    # Goes to the IFD (image file directory) and extracts its data
    ifd_offset    = int.from_bytes(exif_data[10:14], endian)
    ifd0_pos      = 6 + ifd_offset
    if ifd0_pos + 2 > len(exif_data):
        return None

    # IFD stars with 2 bytes of how many entries exist. each entry is 12 bytes
    num_entries   = int.from_bytes(exif_data[ifd0_pos: ifd0_pos + 2], endian)
    entries_start = ifd0_pos + 2


    # Goes through every entry that contains the 0x0131 tag (the EXIF id tag)
    for i in range(num_entries):
        offset = entries_start + i * 12
        if offset + 12 > len(exif_data):
            break

        entry  = exif_data[offset: offset + 12]
        tag_id = int.from_bytes(entry[0:2], endian)

        if tag_id != 0x0131:   # Software tag only
            continue

        # Skips tags not stored in ASCII text
        dtype = int.from_bytes(entry[2:4], endian)
        if dtype != 2:          # must be ASCII
            continue

        # If an exif segment is too long, and the end of it there will be a segment detailing where the rest of the data is

        count      = int.from_bytes(entry[4:8], endian)
        value_raw  = entry[8:12]
        total_bytes = count

        if total_bytes <= 4:
            raw = value_raw[:total_bytes]
        else:
            data_offset = int.from_bytes(value_raw, endian)
            data_pos    = 6 + data_offset
            raw         = exif_data[data_pos: data_pos + total_bytes]

        # ASCII strings have an end tag, we split this section to get the actual text
        return raw.split(b"\x00")[0].decode("ascii", errors="ignore").strip()

    return None


def analyze_metadata(image_path: str) -> dict:
    """
    Analyzes image imetadata and returns a score depending on the likeliness of being ai
    (higher score == more likely)
    """

    # Opens the file
    try:
        with open(image_path, "rb") as f:
            data = f.read()
    except Exception as e:
        return {"score": 0.5, "reason": f"Could not read file: {e}"}

    fmt = _detect_format(data)

    # PNG and WEBP don't embed EXIF the same way skip EXIF checks
    if fmt != "JPEG":
        return {"score": 0.5, "reason": f"{fmt} format — metadata check skipped"}

    # Check for EXIF segment at all
    has_exif = b"\xFF\xE1" in data[:65536]   # only scan file header area

    if not has_exif:
        return {
            "score":  0.35,
            "reason": "JPEG with no EXIF data (mild AI signal)",
        }

    # EXIF present check Software tag
    software = _read_software_tag(data)

    if software is None:
        return {
            "score":  0.20,
            "reason": "EXIF present but no Software tag found",
        }

    software_lower = software.lower()
    ai_hit = next(
        (kw for kw in _AI_SOFTWARE_KEYWORDS if kw in software_lower), None
    )

    if ai_hit:
        return {
            "score":  0.95,
            "reason": f"Software tag identifies AI tool: '{software}'",
        }

    return {
        "score":  0.05,
        "reason": f"Software tag looks like a real tool: '{software}'",
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default path is relative to the project root (imageMetadata/)
    # so we step up two levels from metadata_extract/metadata_scorer.py
    default_path = Path(__file__).resolve().parent.parent / "images" / "testImages" / "aiImageTest.jpg"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)

    result = analyze_metadata(path)
    print(f"Score  : {result['score']}")
    print(f"Reason : {result['reason']}")