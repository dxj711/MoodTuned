"""Shared helpers for MoodTune AI."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from typing import Any

from PIL import Image


def normalize_preferred_genre(value: str) -> str:
    return value.strip() if value else "Auto"


def get_image_hash(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").resize((256, 256)).save(buffer, format="JPEG", quality=85)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def image_to_base64(image: Image.Image, max_size: tuple[int, int] = (360, 360)) -> str:
    thumbnail = image.convert("RGB").copy()
    thumbnail.thumbnail(max_size)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_playlist_pdf(analysis: dict[str, Any], songs: list[dict[str, Any]]) -> bytes:
    """Builds a small PDF; falls back to a valid simple PDF if reportlab is absent."""

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title=analysis.get("playlist_title", "MoodTune Playlist"))
        styles = getSampleStyleSheet()
        story = [
            Paragraph(analysis.get("playlist_title", "MoodTune AI Playlist"), styles["Title"]),
            Spacer(1, 12),
            Paragraph(analysis.get("vibe_description", ""), styles["BodyText"]),
            Spacer(1, 16),
        ]
        for idx, song in enumerate(songs, start=1):
            line = f"{idx}. {song.get('name', 'Song')} - {song.get('artist', 'Artist')} ({song.get('genre_match', 0)}% match)"
            story.append(Paragraph(line, styles["BodyText"]))
            story.append(Paragraph(song.get("why", ""), styles["Italic"]))
            story.append(Spacer(1, 8))
        doc.build(story)
        return buffer.getvalue()
    except Exception:
        title = analysis.get("playlist_title", "MoodTune AI Playlist")
        text = [title, "", analysis.get("vibe_description", ""), ""]
        for idx, song in enumerate(songs, start=1):
            text.append(f"{idx}. {song.get('name', 'Song')} - {song.get('artist', 'Artist')}")
        return _minimal_pdf("\n".join(text))


def _minimal_pdf(text: str) -> bytes:
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = safe_text.splitlines()[:40]
    content_lines = ["BT", "/F1 12 Tf", "72 740 Td"]
    for i, line in enumerate(lines):
        if i:
            content_lines.append("0 -18 Td")
        content_lines.append(f"({line[:90]}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines)
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream.encode('latin-1', errors='ignore'))} >> stream\n{stream}\nendstream endobj",
    ]
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode("latin-1")))
        pdf += obj + "\n"
    xref = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return pdf.encode("latin-1", errors="ignore")


def css() -> str:
    return """
    <style>
    :root {
        --ink: #111827;
        --muted: #64748b;
        --line: rgba(15, 23, 42, 0.12);
        --surface: rgba(255, 255, 255, 0.78);
        --teal: #0f766e;
        --coral: #f97362;
        --gold: #f5b84b;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 32rem),
            radial-gradient(circle at 85% 12%, rgba(249, 115, 98, 0.14), transparent 28rem),
            linear-gradient(135deg, #f8fafc 0%, #eef7f6 48%, #fff8ed 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1320px;
    }
    .hero {
        min-height: 250px;
        display: flex;
        align-items: center;
        padding: 2.5rem 0 1.4rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: clamp(3rem, 7vw, 6rem);
        line-height: 0.95;
        margin: 0.2rem 0 0.8rem;
        letter-spacing: 0;
        color: #0b1220;
    }
    .hero p {
        max-width: 760px;
        font-size: 1.15rem;
        color: #334155;
        margin: 0;
    }
    .eyebrow {
        color: var(--teal);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
    }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 112px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }
    .metric-card span {
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.55rem;
    }
    .metric-card strong {
        display: block;
        font-size: 1.25rem;
        line-height: 1.2;
        color: var(--ink);
    }
    .feature-box {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
    }
    .feature-box > div {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 132px;
    }
    .feature-box b {
        color: var(--teal);
    }
    .feature-box p {
        color: #334155;
        margin-bottom: 0;
    }
    .tag-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.4rem 0 1rem;
    }
    .tag {
        display: inline-flex;
        align-items: center;
        min-height: 30px;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.1);
        border: 1px solid rgba(15, 118, 110, 0.18);
        color: #0f4f49;
        font-size: 0.86rem;
        font-weight: 700;
    }
    .large-tags .tag {
        background: rgba(249, 115, 98, 0.1);
        border-color: rgba(249, 115, 98, 0.2);
        color: #9f3a2e;
    }
    .color-row {
        display: grid;
        grid-template-columns: 34px 1fr;
        align-items: center;
        gap: 0.7rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid var(--line);
    }
    .color-row span {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 1px solid rgba(15, 23, 42, 0.16);
    }
    .color-row small {
        display: block;
        color: var(--muted);
    }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(15, 118, 110, 0.4);
        background: rgba(255, 255, 255, 0.62);
        border-radius: 8px;
    }
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stLinkButton"] a {
        border-radius: 8px;
        font-weight: 800;
    }
    @media (max-width: 780px) {
        .feature-box {
            grid-template-columns: 1fr;
        }
        .hero {
            min-height: 220px;
        }
    }
    </style>
    """


def add_bg_glow() -> None:
    return None
