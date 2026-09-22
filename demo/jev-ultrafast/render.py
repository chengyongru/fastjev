"""Render a FastJev product demo from a verified jev-ultrafast recording."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH, HEIGHT = 1536, 1000
INTRO_MS, OUTRO_MS = 850, 1250
INK = "#0F172A"
MUTED = "#64748B"
PAPER = "#F8FAFC"
LINE = "#CBD5E1"
BLUE = "#2563EB"
PURPLE = "#8B5CF6"
GREEN = "#15803D"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("source", type=Path, help="Verified recording directory")
    result.add_argument("output", type=Path, help="New MP4 output path")
    result.add_argument("--gif", type=Path, help="Optional new GIF output path")
    result.add_argument("--poster", type=Path, help="Optional new PNG poster path")
    return result


def font_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"None of these fonts exists: {candidates}")


REGULAR = font_path(
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
BOLD = font_path(
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
MONO = font_path(
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO if mono else BOLD if bold else REGULAR, size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill: str, *, bold=False, mono=False):
    draw.text(xy, value, font=font(size, bold=bold, mono=mono), fill=fill)


def main() -> None:
    args = parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    outputs = [path.resolve() for path in (args.output, args.gif, args.poster) if path]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    state = json.loads((source / "state.json").read_text(encoding="utf-8"))
    if not state["verification"]["passed"] or state["recording_errors"]:
        raise ValueError("The source recording is not verified")
    frames = [(0, Image.open(source / "frames" / "000000.jpg").convert("RGB"))]
    frames += sorted(
        (int(path.stem), Image.open(path).convert("RGB")) for path in (source / "screencast").glob("*.jpg")
    )
    if len(frames) < 2:
        raise ValueError("The recording has no screencast frames")

    run_end = state["elapsed_ms"]
    total_ms = INTRO_MS + run_end + OUTRO_MS
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_folder = output.parent / "video-frames"
    frame_folder.mkdir(exist_ok=False)
    final_canvas = None

    for index in range(round(total_ms * 30 / 1000)):
        video_ms = round(index * 1000 / 30)
        run_ms = min(run_end, max(0, video_ms - INTRO_MS))
        screenshot = next(image for timestamp, image in reversed(frames) if timestamp <= run_ms)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
        draw = ImageDraw.Draw(canvas)

        draw.polygon([(36, 27), (55, 27), (47, 49), (65, 49), (34, 82), (42, 57), (25, 57)], fill=BLUE)
        text(draw, (79, 27), "Fast", 28, INK, bold=True)
        text(draw, (135, 27), "Jev", 28, PURPLE, bold=True)
        text(draw, (190, 32), "×", 22, MUTED)
        text(draw, (221, 31), "Jev Ultrafast", 23, INK, bold=True)
        draw.rounded_rectangle((1260, 24, 1500, 62), radius=18, fill="#E0E7FF")
        text(draw, (1281, 34), "LOCAL RTX 5090  ·  1×", 14, BLUE, bold=True)

        text(draw, (36, 91), "The official five-action task. Decided locally.", 42, INK, bold=True)
        text(
            draw,
            (38, 151),
            "Qwen3.8-27B EXL3 chooses every operation and target. The browser verifies the result.",
            20,
            MUTED,
        )

        browser_box = (36, 202, 1124, 908)
        draw.rounded_rectangle(browser_box, radius=16, fill="#111827")
        for offset, color in enumerate(("#F87171", "#FBBF24", "#4ADE80")):
            draw.ellipse((57 + 20 * offset, 218, 68 + 20 * offset, 229), fill=color)
        text(draw, (136, 214), "local fixture · structured browser state", 14, "#CBD5E1", mono=True)
        fitted = ImageOps.contain(screenshot, (1050, 644), method=Image.Resampling.LANCZOS)
        x = 55 + (1050 - fitted.width) // 2
        y = 249 + (644 - fitted.height) // 2
        canvas.paste(fitted, (x, y))

        text(draw, (1165, 207), "DECISION LOOP", 14, BLUE, bold=True)
        text(draw, (1162, 237), f"{run_ms / 1000:05.2f}", 54, INK, mono=True)
        text(draw, (1165, 301), "SECONDS · ACTUAL RUN", 13, MUTED, bold=True)
        draw.line((1164, 340, 1499, 340), fill=LINE, width=2)

        history = [item for item in state["history"] if item["executed_ms"] <= run_ms]
        for action_index, item in enumerate(state["history"]):
            y_step = 364 + action_index * 64
            done = item in history
            draw.ellipse((1165, y_step, 1191, y_step + 26), fill=BLUE if done else "#E2E8F0")
            if done:
                draw.line((1172, y_step + 13, 1177, y_step + 18, 1185, y_step + 8), fill="white", width=3)
            label = item["action"]
            for line_index, line in enumerate(textwrap.wrap(label, width=27)[:2]):
                text(
                    draw,
                    (1208, y_step - 9 + line_index * 21),
                    line,
                    16,
                    INK if done else MUTED,
                    bold=done,
                )
            text(
                draw,
                (1208, y_step + 34),
                f"{item['latency_ms']} ms  ·  {item['probability'] * 100:.0f}%",
                14,
                MUTED,
                mono=True,
            )

        finished = video_ms >= INTRO_MS + run_end
        status_y = 699
        draw.rounded_rectangle(
            (1163, status_y, 1500, status_y + 116),
            radius=12,
            fill="#DCFCE7" if finished else "#EEF2FF",
        )
        text(
            draw,
            (1184, status_y + 20),
            "Outcome verified" if finished else "Choose · act · observe",
            21,
            GREEN if finished else INK,
            bold=True,
        )
        text(
            draw,
            (1184, status_y + 58),
            "Casa Flora · Design · free cancellation" if finished else f"{len(history)} of 5 browser actions executed",
            15,
            MUTED,
        )

        latencies = [decision["latency_ms"] for decision in state["decisions"] if decision["elapsed_ms"] <= run_ms]
        median = f"{statistics.median(latencies):.0f} ms" if latencies else "—"
        text(draw, (1165, 826), median, 30, INK, mono=True)
        text(draw, (1165, 868), "median decision request", 15, MUTED)

        progress = 0 if video_ms < INTRO_MS else min(1, run_ms / max(1, run_end))
        draw.line((36, 936, 1500, 936), fill=LINE, width=2)
        draw.line((36, 936, 36 + (1500 - 36) * progress, 936), fill=PURPLE, width=4)
        text(draw, (36, 955), "FastJev System One API · zero output tokens · no TypeSafe API calls", 14, MUTED)
        text(draw, (1012, 955), "Conditional option probabilities are uncalibrated.", 13, MUTED)

        canvas.save(frame_folder / f"{index:04d}.png")
        final_canvas = canvas

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "30",
            "-i",
            str(frame_folder / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    if args.gif:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(output),
                "-vf",
                "fps=12,scale=1152:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop",
                "0",
                str(args.gif),
            ],
            check=True,
        )
    if args.poster:
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        assert final_canvas is not None
        final_canvas.save(args.poster)
    print(f"Rendered {len(frames)} source frames at original run speed: {run_end} ms")


if __name__ == "__main__":
    main()
