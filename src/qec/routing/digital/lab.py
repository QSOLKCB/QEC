# SPDX-License-Identifier: MPL-2.0
"""Deterministic native demo exports. SVG/WAV are presentation, never evidence."""
from __future__ import annotations

import io
import struct
import wave

from .core import validate_digital_receipt


def export_digital_demo(receipt: dict) -> dict:
    validate_digital_receipt(receipt)
    return {"schema": "qec.digital-lab-demo-export.v1", "contract_version": "174.5",
            "canonical_evidence": False, "cryptographic_validation_performed_by_browser": False,
            "presentation_only": True, "source_native_receipt_sha256": receipt["sha256"],
            "label": "OFFLINE SOFTWARE DEMO: no hardware, quantum or carrier claims",
            "accepted": receipt["accepted"], "failure_classes": list(receipt["failure_classes"]),
            "frames": [{"logical_tick": frame["logical_tick"], "symbols": list(frame["symbols"])}
                       for frame in receipt["frames"][:64]],
            "display_frame_limit": 64, "display_truncated": len(receipt["frames"]) > 64,
            "audio_semantics": "1000 samples per frame at 8000 Hz; symbolic square tones; idle is silence"}


def demo_svg(receipt: dict) -> str:
    demo = export_digital_demo(receipt)
    rows = demo["frames"]
    columns = receipt["input"]["frame_schema"]["slot_count"]
    width, height = max(650, 100 + columns * 54), 110 + len(rows) * 28
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="#101a25"/>',
             '<g font-family="monospace" fill="#eaf1f9"><text x="20" y="28">DIGITAL TDM • DEMO ONLY</text>',
             '<text x="20" y="50">Native-validated source; visualisation is not evidence.</text>',
             '<text x="20" y="73">Logical ticks / uint16 slots / IDLE = 256</text>']
    for row, frame in enumerate(rows):
        y = 98 + row * 28
        parts.append(f'<text x="20" y="{y + 16}">{frame["logical_tick"]}</text>')
        for column, symbol in enumerate(frame["symbols"]):
            label = "IDLE" if symbol == 256 else f"{symbol:02x}"
            color = "#304254" if symbol == 256 else "#197965"
            parts.append(f'<rect x="{85 + column * 54}" y="{y}" width="49" height="24" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{89 + column * 54}" y="{y + 16}">{label}</text>')
    parts.append('</g></svg>')
    return "".join(parts) + "\n"


def demo_wav(receipt: dict) -> bytes:
    demo = export_digital_demo(receipt)
    samples = bytearray()
    for frame in demo["frames"]:
        symbols = [symbol for symbol in frame["symbols"] if symbol < 256]
        # Integer-only square wave, with fixed amplitude and explicit silence.
        period = 8 + (symbols[0] % 32) if symbols else 8
        for sample in range(1000):
            value = (6000 if (sample % period) < period // 2 else -6000) if symbols else 0
            samples.extend(struct.pack("<h", value))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(bytes(samples))
    return buffer.getvalue()
