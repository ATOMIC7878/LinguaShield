import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# LinguaShield Graph Generator
# Clean forensic-style graphs for Black Book Section 10
# ============================================================

OUTPUT_DIR = Path("linguashield_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Theme
# ------------------------------------------------------------

BG = "#0b0d0f"
PANEL = "#151719"
GRID = "#292d31"
TEXT = "#f2f2f2"
MUTED = "#8e959a"

ORANGE = "#ff6a00"
ORANGE_LIGHT = "#ff8a3d"
GREEN = "#55d47d"
CYAN = "#47c7d8"
TRACK = "#25292c"

plt.rcParams["font.family"] = "DejaVu Sans"


# ============================================================
# Common save function
# ============================================================

def save_graph(fig, filename):

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=320,
        facecolor=BG,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.25
    )

    plt.close(fig)

    print(f"Created: {output_path}")


# ============================================================
# GRAPH 1
# TEXT + VISUAL VALIDATION
# ============================================================

def create_text_visual_graph():

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    rows = [
        ("English", "Text", 20, ORANGE),
        ("English", "Visual", 20, CYAN),

        ("Hindi", "Text", 20, ORANGE),
        ("Hindi", "Visual", 20, CYAN),

        ("Marathi", "Text", 20, ORANGE),
        ("Marathi", "Visual", 20, CYAN),
    ]

    labels = [
        "English   |   Text",
        "English   |   Visual",
        "Hindi     |   Text",
        "Hindi     |   Visual",
        "Marathi   |   Text",
        "Marathi   |   Visual",
    ]

    values = [row[2] for row in rows]
    colors = [row[3] for row in rows]

    y = np.arange(len(rows))

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11.5, 6.8),
        facecolor=BG
    )

    ax.set_facecolor(PANEL)

    # --------------------------------------------------------
    # Background tracks
    # --------------------------------------------------------

    ax.barh(
        y,
        [20] * len(values),
        height=0.48,
        color=TRACK,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Actual result bars
    # --------------------------------------------------------

    bars = ax.barh(
        y,
        values,
        height=0.48,
        color=colors,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Direct labels
    # --------------------------------------------------------

    for bar, value in zip(bars, values):

        ax.text(
            value - 0.35,
            bar.get_y() + bar.get_height() / 2,
            "20/20 PASS",
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#ffffff"
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(0, 20.8)

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=10.5,
        color=TEXT
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Successful Test Cases",
        fontsize=10,
        color=MUTED,
        labelpad=10
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "TEXT AND VISUAL VALIDATION",
        loc="left",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        pad=20
    )

    ax.text(
        0,
        1.015,
        "Recorded multilingual validation results",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
        va="bottom"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.set_xticks([0, 5, 10, 15, 20])

    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.8,
        alpha=0.8
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Borders
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(GRID)

    ax.tick_params(
        axis="x",
        colors=MUTED,
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        length=0
    )

    # --------------------------------------------------------
    # Small module indicators
    # --------------------------------------------------------

    for i, color in enumerate(colors):

        ax.scatter(
            -0.8,
            i,
            s=65,
            color=color,
            clip_on=False,
            zorder=10
        )

    # Leave room for the indicators
    ax.set_xlim(-1.2, 20.8)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_graph(
        fig,
        "text_visual_validation.png"
    )


# ============================================================
# GRAPH 2
# ACOUSTIC AI SCORES
# ============================================================

def create_acoustic_graph():

    labels = [
        "Digital Arrest",
        "Stock / Crypto",
        "Family Voice Clone",
        "Remote Access",
        "CEO Wire Scam",
        "Cloned Bank",
        "Gaming Nitro"
    ]

    scores = np.array([
        65,
        98,
        90,
        75,
        98,
        90,
        98
    ])

    fig, ax = plt.subplots(
        figsize=(11.5, 6.8),
        facecolor=BG
    )

    ax.set_facecolor(PANEL)

    x = np.arange(len(labels))

    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    bars = ax.bar(
        x,
        scores,
        width=0.60,
        color=ORANGE
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bar, score in zip(bars, scores):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 2.2,
            f"{score}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=TEXT
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_ylim(0, 110)

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right",
        fontsize=9.5,
        color=TEXT
    )

    ax.set_ylabel(
        "Recorded AI Score (%)",
        fontsize=10,
        color=MUTED,
        labelpad=10
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "ACOUSTIC AI SCORE RESULTS",
        loc="left",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        pad=20
    )

    ax.text(
        0,
        1.015,
        "Recorded scores from tested synthetic-voice samples",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
        va="bottom"
    )

    # --------------------------------------------------------
    # Reference line
    # --------------------------------------------------------

    ax.axhline(
        75,
        color=ORANGE_LIGHT,
        linewidth=1,
        linestyle="--",
        alpha=0.4
    )

    # Put the label safely inside the plot
    ax.text(
        len(labels) - 0.05,
        77,
        "75%",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        alpha=0.75
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Borders
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(GRID)

    ax.tick_params(
        axis="y",
        colors=MUTED,
        labelsize=9
    )

    ax.tick_params(
        axis="x",
        length=0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_graph(
        fig,
        "acoustic_ai_scores.png"
    )


# ============================================================
# GRAPH 3
# BIOMETRIC SPEAKER SIMILARITY
# ============================================================

def create_biometric_graph():

    labels = [
        "Same Voice",
        "V7 vs V5",
        "V7 vs V6",
        "V5 vs V6",
        "Same AI Voice",
        "V8 vs V7",
        "V7.1 vs V7"
    ]

    values = np.array([
        100,
        21.6,
        17.0,
        34.4,
        89.4,
        27.1,
        76.2
    ])

    colors = [
        GREEN,
        ORANGE,
        ORANGE,
        ORANGE,
        ORANGE_LIGHT,
        ORANGE,
        CYAN
    ]

    y = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(11.5, 6.8),
        facecolor=BG
    )

    ax.set_facecolor(PANEL)

    # --------------------------------------------------------
    # Background tracks
    # --------------------------------------------------------

    ax.barh(
        y,
        [100] * len(values),
        height=0.46,
        color=TRACK,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Actual similarity bars
    # --------------------------------------------------------

    bars = ax.barh(
        y,
        values,
        height=0.46,
        color=colors,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Direct value labels
    # --------------------------------------------------------

    for bar, value in zip(bars, values):

        if float(value).is_integer():
            text_value = f"{int(value)}%"
        else:
            text_value = f"{value:.1f}%"

        # Put label inside when the bar is long,
        # outside when the bar is short.
        if value >= 55:

            ax.text(
                value - 1.2,
                bar.get_y() + bar.get_height() / 2,
                text_value,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#ffffff"
            )

        else:

            ax.text(
                value + 1.2,
                bar.get_y() + bar.get_height() / 2,
                text_value,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=TEXT
            )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(0, 108)

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=10.5,
        color=TEXT
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Similarity / Match (%)",
        fontsize=10,
        color=MUTED,
        labelpad=10
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "BIOMETRIC SPEAKER SIMILARITY",
        loc="left",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        pad=20
    )

    ax.text(
        0,
        1.015,
        "Recorded similarity values from tested voice comparisons",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
        va="bottom"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.8,
        alpha=0.8
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Borders
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(GRID)

    ax.tick_params(
        axis="x",
        colors=MUTED,
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        length=0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_graph(
        fig,
        "biometric_speaker_similarity.png"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_text_visual_graph()
    create_acoustic_graph()
    create_biometric_graph()

    print("\nAll LinguaShield graphs generated successfully.")