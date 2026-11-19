"""Small 2D gamepad overlay for the MuJoCo viewer."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT = 0x0001, 0x0002, 0x0004, 0x0008
START, BACK = 0x0010, 0x0020
LB, RB = 0x0100, 0x0200
A, B, X, Y = 0x1000, 0x2000, 0x4000, 0x8000


def draw_gamepad(pad: Any, width: int = 320, height: int = 200) -> np.ndarray:
    """Draw sticks, triggers, D-pad and ABXY state."""
    image = Image.new("RGB", (width, height), (16, 18, 22))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle(
        (10, 18, width - 10, height - 10),
        radius=46,
        fill=(28, 31, 38),
        outline=(92, 104, 120),
        width=2,
    )
    draw.text((18, 6), "GAMEPAD", fill=(180, 190, 205), font=font)
    if pad is None:
        draw.text(
            (width // 2, height // 2),
            "NO PAD",
            fill=(230, 90, 90),
            font=font,
            anchor="mm",
        )
        return np.asarray(image)

    def stick(cx: int, cy: int, x: float, y: float, label: str) -> None:
        r = 24
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(150, 160, 175), width=2)
        draw.line((cx - r, cy, cx + r, cy), fill=(55, 60, 70))
        draw.line((cx, cy - r, cx, cy + r), fill=(55, 60, 70))
        dx = int(max(-1.0, min(1.0, x)) * r * 0.7)
        dy = int(max(-1.0, min(1.0, -y)) * r * 0.7)
        draw.ellipse(
            (cx + dx - 6, cy + dy - 6, cx + dx + 6, cy + dy + 6),
            fill=(80, 200, 255),
        )
        draw.text((cx, cy + r + 7), label, fill=(150, 160, 175), font=font, anchor="ma")

    def bar(x: int, y: int, value: float, label: str) -> None:
        w = 58
        draw.rounded_rectangle(
            (x, y, x + w, y + 9),
            radius=4,
            fill=(45, 50, 60),
            outline=(90, 100, 115),
        )
        fill = int(w * max(0.0, min(1.0, value)))
        if fill:
            draw.rectangle((x + 1, y + 1, x + fill, y + 8), fill=(80, 200, 255))
        draw.text((x + w // 2, y - 3), label, fill=(150, 160, 175), font=font, anchor="ms")

    bar(35, 44, pad.lt, "LT")
    bar(227, 44, pad.rt, "RT")
    draw.rounded_rectangle(
        (30, 20, 75, 32),
        radius=4,
        fill=(80, 200, 255) if pad.buttons & LB else (55, 60, 70),
    )
    draw.text((52, 21), "LB", fill=(230, 235, 240), font=font, anchor="ma")
    draw.rounded_rectangle(
        (245, 20, 290, 32),
        radius=4,
        fill=(80, 200, 255) if pad.buttons & RB else (55, 60, 70),
    )
    draw.text((267, 21), "RB", fill=(230, 235, 240), font=font, anchor="ma")

    stick(82, 128, pad.lx, pad.ly, "L")
    stick(238, 128, pad.rx, pad.ry, "R")

    # D-pad.
    cx, cy = 82, 78
    color = (80, 200, 255)
    base = (55, 60, 70)
    draw.rectangle((cx - 6, cy - 18, cx + 6, cy - 5), fill=color if pad.buttons & DPAD_UP else base)
    draw.rectangle(
        (cx - 6, cy + 5, cx + 6, cy + 18), fill=color if pad.buttons & DPAD_DOWN else base
    )
    draw.rectangle(
        (cx - 18, cy - 6, cx - 5, cy + 6), fill=color if pad.buttons & DPAD_LEFT else base
    )
    draw.rectangle(
        (cx + 5, cy - 6, cx + 18, cy + 6), fill=color if pad.buttons & DPAD_RIGHT else base
    )

    # ABXY diamond.
    def button(x: int, y: int, mask: int, label: str, tint: tuple[int, int, int]) -> None:
        active = bool(pad.buttons & mask)
        draw.ellipse(
            (x - 9, y - 9, x + 9, y + 9),
            fill=tint if active else (55, 60, 70),
            outline=(210, 215, 225),
        )
        draw.text((x, y), label, fill=(15, 18, 22), font=font, anchor="mm")

    button(238, 48, Y, "Y", (250, 210, 70))
    button(238, 86, A, "A", (80, 220, 120))
    button(220, 67, X, "X", (100, 160, 255))
    button(256, 67, B, "B", (255, 100, 100))
    return np.asarray(image)


class GamepadWindow:
    """Separate Tk window so the controller panel does not cover the scene."""

    def __init__(self) -> None:
        import tkinter as tk

        from PIL import ImageTk

        self.tk = tk
        self.ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title("Alicia gamepad")
        self.root.resizable(False, False)
        self.root.configure(bg="#101216")
        self.root.attributes("-topmost", True)
        self.label = tk.Label(self.root, bd=0, bg="#101216")
        self.label.pack()
        self.photo = None
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()
        width, height = 320, 200
        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"{width}x{height}+{max(20, screen_w - width - 36)}+90")

    def update(self, pad: Any) -> None:
        if self.closed:
            return
        try:
            image = Image.fromarray(draw_gamepad(pad))
            self.photo = self.ImageTk.PhotoImage(image)
            self.label.configure(image=self.photo)
            self.root.update_idletasks()
            self.root.update()
        except Exception:  # noqa: BLE001
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass
