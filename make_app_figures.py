from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ASSET_DIR = Path("app/assets")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

GREEN = "#2E7D32"
LIGHT_GREEN = "#E8F5E9"
DARK = "#1F2933"
GRAY = "#6B7280"
BLUE = "#E3F2FD"
ORANGE = "#FFF3E0"


def add_box(ax, x, y, w, h, text, facecolor="white", edgecolor=GREEN, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=DARK, wrap=True)


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color=GRAY,
    )
    ax.add_patch(arrow)


# Figure 1: Product workflow
fig, ax = plt.subplots(figsize=(15, 4.8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 4.8)
ax.axis("off")

steps = [
    ("Retail sales\n+ stock data", LIGHT_GREEN),
    ("Feature\nengineering", "white"),
    ("Latent demand\nrecovery", BLUE),
    ("Lost-sales\nestimation", ORANGE),
    ("Forecast\nmodel", BLUE),
    ("FastAPI\nbackend", "white"),
    ("Streamlit\napp", LIGHT_GREEN),
    ("Business\ndecision support", ORANGE),
]

x_positions = np.linspace(0.35, 12.9, len(steps))
for i, (x, item) in enumerate(zip(x_positions, steps)):
    label, color = item
    add_box(ax, x, 1.65, 1.55, 1.05, label, facecolor=color, fontsize=9.5)
    if i < len(steps) - 1:
        add_arrow(ax, x + 1.55, 2.175, x_positions[i + 1], 2.175)

ax.text(0.35, 4.15, "Fresh Retail Copilot", fontsize=18, fontweight="bold", color=GREEN)
ax.text(0.35, 3.78, "End-to-end stockout-aware demand forecasting workflow", fontsize=11, color=GRAY)
plt.tight_layout()
plt.savefig(ASSET_DIR / "retail_flow.png", dpi=220, bbox_inches="tight")
plt.close()


# Figure 2: Stockout concept
days = np.arange(1, 31)
true_demand = 18 + 4 * np.sin(days / 3) + 0.25 * days
availability = np.ones_like(days, dtype=float)
availability[11:17] = 0.35
availability[23:27] = 0.55
observed_sales = true_demand * availability

fig, ax = plt.subplots(figsize=(12.5, 5.4))
ax.plot(days, true_demand, linewidth=3, label="Estimated true demand", color=GREEN)
ax.plot(days, observed_sales, linewidth=3, linestyle="--", label="Observed sales", color="#455A64")
ax.fill_between(days, observed_sales, true_demand, where=true_demand > observed_sales, alpha=0.25, color="#EF6C00", label="Potential lost sales")

ax.axvspan(12, 17, color="#EF6C00", alpha=0.08)
ax.axvspan(24, 27, color="#EF6C00", alpha=0.08)
ax.text(12.1, max(true_demand) - 1, "stockout window", fontsize=9, color="#BF360C")
ax.text(24.1, max(true_demand) - 2.5, "partial availability", fontsize=9, color="#BF360C")

ax.set_title("Observed Sales Can Understate True Demand", fontsize=16, pad=14, color=DARK, fontweight="bold")
ax.set_xlabel("Day", color=DARK)
ax.set_ylabel("Units", color=DARK)
ax.legend(frameon=True)
ax.grid(True, alpha=0.22)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(ASSET_DIR / "stockout_concept.png", dpi=220, bbox_inches="tight")
plt.close()


# Figure 3: Cloud architecture
fig, ax = plt.subplots(figsize=(12.5, 5.6))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 5.6)
ax.axis("off")

add_box(ax, 0.6, 2.25, 1.8, 1.0, "User\nBrowser", facecolor=LIGHT_GREEN, fontsize=10)
add_box(ax, 3.1, 2.25, 2.0, 1.0, "Streamlit\nFrontend", facecolor="white", fontsize=10)
add_box(ax, 5.9, 2.25, 2.0, 1.0, "FastAPI\nBackend", facecolor=BLUE, fontsize=10)
add_box(ax, 8.7, 3.35, 2.1, 1.0, "Model\nArtifact", facecolor="white", fontsize=10)
add_box(ax, 8.7, 1.15, 2.1, 1.0, "Processed\nRetail Data", facecolor=ORANGE, fontsize=10)

add_arrow(ax, 2.4, 2.75, 3.1, 2.75)
add_arrow(ax, 5.1, 2.75, 5.9, 2.75)
add_arrow(ax, 7.9, 2.95, 8.7, 3.85)
add_arrow(ax, 7.9, 2.55, 8.7, 1.65)

ax.text(3.55, 3.58, "scenario selection", ha="center", fontsize=9, color=GRAY)
ax.text(6.92, 3.58, "feature payload", ha="center", fontsize=9, color=GRAY)
ax.text(6.92, 1.65, "predicted demand", ha="center", fontsize=9, color=GRAY)

ax.text(0.6, 5.05, "Frontend + Backend Architecture", fontsize=18, fontweight="bold", color=GREEN)
ax.text(0.6, 4.68, "The frontend remains lightweight while the model is served through an API layer.", fontsize=11, color=GRAY)

plt.tight_layout()
plt.savefig(ASSET_DIR / "cloud_architecture.png", dpi=220, bbox_inches="tight")
plt.close()

print("Updated visual assets in app/assets/")
