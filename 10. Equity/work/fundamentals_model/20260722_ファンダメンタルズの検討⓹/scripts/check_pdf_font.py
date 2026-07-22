from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stock_scoring_model.config_loader import load_config
from stock_scoring_model.font_support import setup_japanese_matplotlib_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="日本語PDFフォントの検出・埋め込み確認")
    parser.add_argument("--config", default="config/model_config.py")
    parser.add_argument("--output", default="outputs/japanese_font_check.pdf")
    args = parser.parse_args()

    config, root = load_config(Path(args.config).resolve())
    family, path = setup_japanese_matplotlib_from_config(config, root)
    output = Path(root) / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        lines = [
            "日本語PDFフォント確認",
            "国別Aggregate FactorScore推移",
            "バリュー・モメンタム・クオリティ・低ボラティリティ",
            "Q5-Q1累積リターン / ローリングRank IC / 第3層回帰係数",
            f"選択フォント: {family}",
            f"フォントパス: {path}",
        ]
        for i, line in enumerate(lines):
            ax.text(0.06, 0.90 - i * 0.12, line, fontsize=20 if i == 0 else 14, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Japanese font family: {family}")
    print(f"Japanese font path:   {path}")
    print(f"Check PDF:             {output}")


if __name__ == "__main__":
    main()
