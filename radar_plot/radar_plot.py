import math
import matplotlib.pyplot as plt
import pandas as pd
from plot_utils import ComplexRadar, get_ranges

yellow = "#FFD700"
dark_yellow = "#FF8C00"
dark_green = "#006400"
dark_blue = "#00008B"
red = "#FF0000"
black = "#000000"
grey = "#808080"

format_cfg = {
    "rad_ln_args": {"visible": True},
    "outer_ring": {"visible": True},
    "angle_ln_args": {"visible": True},
    "rgrid_tick_lbls_args": {"fontsize": 9},
    "theta_tick_lbls": {"fontsize": 12},
    "theta_tick_lbls_pad": 11,
}


d = {
    'time': [10, 7, 5, 5, 5, 5, 0],
    'accuracy': [0.975, 0.974, 0.971, 0.965, 0.957, 0.951, 0.750],
    'energy': [0.525, 0.178, 0.132, 0.133, 0.129, 0.160, 0],
    'params': [36.5, 20.5, 10.4, 7.0, 5.4, 4.3, 0]
}
model_label = ['WRN', 'WRN-MPS-2', 'WRN-MPS-4', 'WRN-MPS-6', 'WRN-MPS-8', 'WRN-MPS-10']
df = pd.DataFrame(d)

ranges = get_ranges(df)

fig = plt.figure(figsize=(6,6))
radar = ComplexRadar(fig, df.columns, ranges, n_ring_levels=5, show_scales=True, format_cfg=format_cfg)
radar.plot(df.iloc[0].values.tolist(), color=dark_blue, linewidth=0.8, label=model_label[0])
radar.fill(df.iloc[0].values.tolist(), alpha=0.4, color=dark_blue)

radar.plot(df.iloc[1].values.tolist(), color=dark_green, linewidth=0.8, label=model_label[1])
radar.fill(df.iloc[1].values.tolist(), alpha=0.4, color=dark_green)

radar.plot(df.iloc[2].values.tolist(), color=yellow, linewidth=0.8, label=model_label[2])
radar.fill(df.iloc[2].values.tolist(), alpha=0.4, color=yellow)

radar.plot(df.iloc[4].values.tolist(), color=red, linewidth=0.8, label=model_label[4])
radar.fill(df.iloc[4].values.tolist(), alpha=0.4, color=red)

radar.use_legend(loc='lower left', bbox_to_anchor=(0.15, -0.15),ncol=radar.plot_counter)

plt.savefig("./result.png", dpi=300, bbox_inches="tight")