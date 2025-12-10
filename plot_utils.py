# import polars as pl
# import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
from matplotlib.figure import figaspect

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.size": "21",
#     "font.family": "serif",
#     # "font.serif": ["Palatino"],
#     "legend.title_fontsize": "18",
# })

# density=5
# hatches=[hatch * density for hatch in ['/','.','x','\\','+','-','|']]
# colors=['#5BB8D7','#57A86B','#A8A857','#6E4587','#ADEBCC','#EBDCAD']

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.axes import Axes
import numpy as np

# --- Global Style Settings ---
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 24, # Slightly larger for readability
    "font.family": "serif",
    "legend.title_fontsize": 20,
    "legend.fontsize": 18,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})


DENSITY = 4 
HATCHES = [h * DENSITY for h in ['/', 'x', '.', '\\', '+', 'o', '|']]
COLORS = ['#5BB8D7', '#57A86B', '#A8A857', '#6E4587', '#ADEBCC', '#EBDCAD', '#F0A3FF']

def parity_plot(
    df: pl.DataFrame,
    x: str,
    y: str,
    hue: str,   # Column for Color/Hatch (Model)
    style: str, # Column for Shape (Phase)
    ax: Axes = None,
    ms: int = 180
):
    if ax is None:
        # Square figure aspect
        fig, ax = plt.subplots(figsize=(7, 7))

    # 1. Setup Mappings
    
    # Sort uniques to ensure consistent coloring across plots
    unique_hues = df.select(hue).unique().sort(hue)[hue].to_list()
    unique_styles = df.select(style).unique().sort(style)[style].to_list()

    # Create Color/Hatch Map
    hue_map = {
        h: {
            "color": COLORS[i % len(COLORS)],
            "hatch": HATCHES[i % len(HATCHES)]
        } 
        for i, h in enumerate(unique_hues)
    }

    # Create Marker Map (Enforcing your specific requirement)
    # Default fallbacks provided if data isn't exactly "train"/"inference"
    style_map_defaults = {'inference': 's', 'train': '^'} 
    fallback_markers = ['o', 'D', 'v', '<', '>']
    style_map = {}
    
    marker_idx = 0
    for s_val in unique_styles:
        # Normalize string for matching (e.g. "Inference" -> "inference")
        s_lower = s_val.lower() 
        if "inference" in s_lower:
            style_map[s_val] = 's' # Square
        elif "train" in s_lower:
            style_map[s_val] = '^' # Triangle
        else:
            style_map[s_val] = fallback_markers[marker_idx % len(fallback_markers)]
            marker_idx += 1

    # 2. Plotting Loop
    # Group by both Hue and Style to plot specific points
    for (h_val, s_val), group in df.group_by([hue, style]):
        props = hue_map[h_val]
        marker = style_map[s_val]
        
        ax.scatter(
            group[x],
            group[y],
            s=ms,
            marker=marker,
            facecolor='white',  # White background so hatches show clearly
            edgecolor=props['color'],
            hatch=props['hatch'],
            zorder=3,
            linewidth=1.5,
            alpha=1.0
        )

    # 3. Diagonal Line (y=x)
    all_vals = df.select(pl.col(x), pl.col(y)).to_numpy()
    low, high = np.min(all_vals), np.max(all_vals)
    pad = (high - low) * 0.1
    limit_low = max(0, low - pad) # Assuming power >= 0
    limit_high = high + pad
    
    ax.plot([limit_low, limit_high], [limit_low, limit_high], 
            color='gray', linestyle='--', linewidth=2, zorder=2)

    # 4. Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.grid(linestyle='--', zorder=0, alpha=0.5)

    # 5. Construct Custom Legends
    
    # Legend 1: Models (Color + Hatch)
    hue_handles = []
    for h_val in unique_hues:
        props = hue_map[h_val]
        patch = mpatches.Patch(
            facecolor='white',
            edgecolor=props['color'],
            hatch=props['hatch'],
            label=h_val,
            linewidth=1.5
        )
        hue_handles.append(patch)

    # Legend 2: Phase (Shape Only)
    style_handles = []
    for s_val in unique_styles:
        m = style_map[s_val]
        # Create a Line2D marker with white face and black edge
        handle = mlines.Line2D(
            [], [], 
            color='white', # Hide the line 
            marker=m, 
            markeredgecolor='black', # Black outline for shape legend
            markerfacecolor='white',
            markersize=12, 
            markeredgewidth=1.5,
            label=s_val, 
            linestyle='None'
        )
        style_handles.append(handle)

    return ax, hue_handles, style_handles

def barplot(df: pl.DataFrame, x: str, y: str, hue: str = None, ylabel: str = None, ndigits: int = 2, ax: Axes = None, w = None):
    if ax is None:
        w, h = figaspect(0.4/0.9)
        fig, ax = plt.subplots(figsize=(w, h))
    hue_expr = pl.lit("") if hue is None else pl.col(hue)
    ylabel_expr = None if ylabel is None else pl.col(ylabel)
    df = df.select(x=pl.col(x), y=pl.col(y), hue=hue_expr, ylabel=ylabel_expr)
    plot_df = (
        df.join(
            df.select("hue").unique(maintain_order=True).with_row_index(name="order"),
            on="hue",
        )
        .with_row_index()
        .sort(pl.col("index").min().over("x"), "index")
        .with_columns(pl.cum_count("x").add(pl.col("x").ne_missing(pl.col("x").shift(1)).cum_sum() * 0.5 - 1.5).alias("xloc"))
        .select(
            pl.col("xloc") * w / (pl.col("xloc").max() + 1),
            pl.col("y").alias("height"),
            (w / (pl.col("xloc").max() + 1)).alias("width"),
            pl.col("hue").alias("label"),
            pl.col("order").map_elements(lambda x: hatches[x % len(hatches)], return_dtype=str).alias("hatch"),
            pl.col("order").map_elements(lambda x: colors[x % len(colors)], return_dtype=str).alias("edgecolor"),
            pl.col("x").alias("xlabel"),
            "ylabel",
        )
    )
        
    for (label,), df in plot_df.partition_by("label", as_dict=True).items():
        bars = ax.bar(
            df["xloc"], 
            df["height"], 
            df["width"], 
            label=label, 
            hatch=df["hatch"], 
            edgecolor=df["edgecolor"], 
            zorder=3, 
            fill=False,
        )
        if ylabel is None:
            continue
        for i, bar in enumerate(bars):
            label_val = df["ylabel"][i]
            va = 'bottom' if label_val >= 0 else 'top'
            ax.annotate(
                f"{label_val:.{ndigits}f}",
                (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                ha='center', va=va,
                fontsize=15, color='black',
            )

    xticks_df = plot_df.group_by("xlabel").agg(pl.col("xloc").mean())
    ax.set_xticks(xticks_df["xloc"], xticks_df["xlabel"])
    return ax