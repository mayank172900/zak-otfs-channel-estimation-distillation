from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .compat import strict_zip


def curve_plot_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    style_col = None
    if "modulation" in df.columns and df["modulation"].nunique() > 1:
        style_col = "modulation"
    return "method", style_col


def save_curve_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    path: Path,
    logy: bool = True,
    style_col: str | None = None,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plot_kwargs = {"data": df, "x": x_col, "y": y_col, "hue": hue_col}
    if style_col is None:
        plot_kwargs["marker"] = "o"
    else:
        plot_kwargs["style"] = style_col
        plot_kwargs["markers"] = True
        plot_kwargs["dashes"] = True
    sns.lineplot(**plot_kwargs)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_heatmaps(images: list[np.ndarray], titles: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    for ax, image, title in strict_zip(axes, images, titles):
        im = ax.imshow(np.abs(image), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
