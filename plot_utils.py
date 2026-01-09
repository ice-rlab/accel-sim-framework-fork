import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.figure import figaspect
from matplotlib.axes import Axes

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 18, "legend.title_fontsize": 16
})

def plot_bar(df: pl.DataFrame, x: str, y: str, hue: str = None, ylabel: str = None, ndigits: int = 2, ax: Axes = None, w=None):
    density = 4
    hatches = [h * density for h in ['/', 'x', '.', '\\', '+', 'o', '|']]
    colors = ['#5BB8D7', '#57A86B', '#A8A857', '#6E4587', '#ADEBCC', '#EBDCAD', '#F0A3FF']
    
    if ax is None:
        default_w, default_h = figaspect(0.4/0.9)
        w = w if w is not None else default_w
        fig, ax = plt.subplots(figsize=(w, default_h))
    else:
        w = w if w is not None else 10 

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
        .with_columns(
            pl.cum_count("x").add(
                pl.col("x").ne_missing(pl.col("x").shift(1)).cum_sum() * 0.5 - 1.5
            ).alias("xloc_raw")
        )
        .select(
            (pl.col("xloc_raw") * w / (pl.col("xloc_raw").max() + 1)).alias("xloc"),
            pl.col("y").alias("height"),
            (w / (pl.col("xloc_raw").max() + 1)).alias("width"),
            pl.col("hue").alias("label"),
            pl.col("order").map_elements(lambda x: hatches[x % len(hatches)], return_dtype=str).alias("hatch"),
            pl.col("order").map_elements(lambda x: colors[x % len(colors)], return_dtype=str).alias("edgecolor"),
            pl.col("x").alias("xlabel"),
            "ylabel",
        )
    )
        
    for (label,), group in plot_df.partition_by("label", as_dict=True).items():
        bars = ax.bar(
            group["xloc"], 
            group["height"], 
            group["width"], 
            label=label, 
            hatch=group["hatch"], 
            edgecolor=group["edgecolor"], 
            zorder=3, 
            fill=False,
        )
        
        if ylabel is None:
            continue
            
        for i, bar in enumerate(bars):
            label_val = group["ylabel"][i]
            if label_val is None: continue
            
            va = 'bottom' if label_val >= 0 else 'top'
            ax.annotate(
                f"{label_val:.{ndigits}f}",
                (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                ha='center', va=va,
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=10, color='black',
            )

    xticks_df = plot_df.group_by("xlabel", maintain_order=True).agg(pl.col("xloc").mean())
    ax.set_xticks(xticks_df["xloc"], xticks_df["xlabel"], rotation=45, ha='right')

    if hue is not None:
        ax.legend()
        
    return ax

def plot_xy(df, x, y, x_label, y_label, convert_fn=None, legend_loc='lower right', save=False, fig_num=1):
    if convert_fn:
        df = df.with_columns(
            pl.col(x).map_elements(convert_fn, return_dtype=pl.Float64).alias(x),
            pl.col(y).map_elements(convert_fn, return_dtype=pl.Float64).alias(y)
        )

    plot_df = df.with_columns(
        pl.col("benchmark").str.split("-").list.first().alias("model"),
        pl.col("benchmark").str.split("-").list.last().alias("phase")
    )

    models = plot_df["model"].unique().sort().to_list()
    colors = ['#5BB8D7','#57A86B','#A8A857','#6E4587','#ADEBCC','#EBDCAD']
    model_map = {m: c for m, c in zip(models, colors)}
    phase_map = {'inference': 's', 'train': '^'} 

    fig, ax = plt.subplots(figsize=(6, 6))
    
    for (model, phase), data in plot_df.group_by(["model", "phase"]):
        ax.scatter(
            data[x], data[y],
            c=model_map.get(model, 'gray'), marker=phase_map.get(phase, 'o'),
            s=150, edgecolors='black', alpha=0.9, zorder=3
        )

    limit = max(df[x].max(), df[y].max()) * 1.1
    
    ax.plot([0, limit], [0, limit], "gray", ls="--", zorder=1)
    
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal', adjustable='box')
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6)) 
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, ls='--', alpha=0.5)

    handles = [mpatches.Patch(color='none', label=r'$\bf{Model}$')] 
    handles += [mpatches.Patch(facecolor=c, edgecolor='k', label=m) for m, c in model_map.items()]
    
    handles += [mpatches.Patch(color='none', label=''), 
                mpatches.Patch(color='none', label=r'$\bf{Phase}$')] 
    handles += [mlines.Line2D([], [], color='w', marker=m, markeredgecolor='k', ms=10, label=p) 
                for p, m in phase_map.items() if p in plot_df["phase"].unique()]

    ax.legend(handles=handles, loc=legend_loc, frameon=True, fontsize=12)
    
    if save:
        plt.savefig(f'Figure_{fig_num}.pdf', bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()