import numpy as np
import textwrap
import pandas as pd

class ComplexRadar:
    """
    Create a complex radar chart with different scales for each variable
    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if we the ranges for each variable are plotted
    format_cfg: dict, defaults to None
        A dictionary with formatting configurations
    """

    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True, format_cfg=None):

        # Default formatting
        self.format_cfg = {
            # Axes
            # https://matplotlib.org/stable/api/figure_api.html
            "axes_args": {},
            # Tick labels on the scales
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rgrids.html
            "rgrid_tick_lbls_args": {"fontsize": 8},
            # Radial (circle) lines
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
            "rad_ln_args": {},
            # Angle lines
            # https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
            "angle_ln_args": {},
            # Include last value (endpoint) on scale
            "incl_endpoint": True,
            # Variable labels (ThetaTickLabel)
            "theta_tick_lbls": {"va": "top", "ha": "center"},
            "theta_tick_lbls_txt_wrap": 15,
            "theta_tick_lbls_brk_lng_wrds": False,
            "theta_tick_lbls_pad": 25,
            # Outer ring
            # https://matplotlib.org/stable/api/spines_api.html
            "outer_ring": {"visible": True, "color": "#d6d6d6"},
        }

        if format_cfg is not None:
            self.format_cfg = {
                k: (format_cfg[k]) if k in format_cfg.keys() else (self.format_cfg[k]) for k in self.format_cfg.keys()
            }

            # Calculate angles and create for each variable an axes
        # Consider here the trick with having the first axes element twice (len+1)
        angles = np.arange(0, 360, 360.0 / len(variables))
        axes = [
            fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True, label="axes{}".format(i), **self.format_cfg["axes_args"])
            for i in range(len(variables) + 1)
        ]

        # Ensure clockwise rotation (first variable at the top N)
        for ax in axes:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)

        # Writing the ranges on each axes
        for i, ax in enumerate(axes):

            # Here we do the trick by repeating the first iteration
            j = 0 if (i == 0 or i == 1) else i - 1
            ax.set_ylim(*ranges[j])

            # Set endpoint to True if you like to have values right before the last circle
            grid = np.linspace(*ranges[j], num=n_ring_levels, endpoint=self.format_cfg["incl_endpoint"])
            gridlabel = [
                f"{x:.2f}".format(round(x, 0)) if ranges[j][1] < 3 * 1e6 else f"{x/1e6:.2f}".format(round(x, 0))
                for x in grid
            ]
            gridlabel[0] = ""  # remove values from the center
            lines, labels = ax.set_rgrids(
                grid, labels=gridlabel, angle=angles[j], **self.format_cfg["rgrid_tick_lbls_args"]
            )

            ax.set_ylim(*ranges[j])
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)

            if show_scales == False:
                ax.set_yticklabels([])

        # Set all axes except the first one unvisible
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)

        # Setting the attributes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0

        # Draw (inner) circles and lines
        self.ax.yaxis.grid(**self.format_cfg["rad_ln_args"])
        # Draw outer circle
        self.ax.spines["polar"].set(**self.format_cfg["outer_ring"])
        # Draw angle lines
        self.ax.xaxis.grid(**self.format_cfg["angle_ln_args"])

        # ax1 is the duplicate of axes[0] (self.ax)
        # Remove everything from ax1 except the plot itself
        self.ax1.axis("off")
        self.ax1.set_zorder(9)

        # Create the outer labels for each variable
        l, text = self.ax.set_thetagrids(angles, labels=variables)

        # Beautify them
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        labels = [
            "\n".join(
                textwrap.wrap(
                    l,
                    self.format_cfg["theta_tick_lbls_txt_wrap"],
                    break_long_words=self.format_cfg["theta_tick_lbls_brk_lng_wrds"],
                )
            )
            for l in labels
        ]
        labels = ["$T$(s)", "$P_{v}$", "$E$(kWh)", r"$|\theta|$(M)"]
        self.ax.set_xticklabels(labels, **self.format_cfg["theta_tick_lbls"])

        for t, a in zip(self.ax.get_xticklabels(), angles):
            if a == 0:
                t.set_ha("center")
            elif a > 0 and a < 180:
                t.set_ha("left")
            elif a == 180:
                t.set_ha("center")
            else:
                t.set_ha("right")

        self.ax.tick_params(axis="both", pad=self.format_cfg["theta_tick_lbls_pad"])

    def _scale_data(self, data, ranges):
        """Scales data[1:] to ranges[0]"""
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
        return sdata

    def plot(self, data, *args, **kwargs):
        """Plots a line"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        self.plot_counter = self.plot_counter + 1

    def fill(self, data, *args, **kwargs):
        """Plots an area"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        self.ax1.legend(*args, **kwargs)

    def set_title(self, title, pad=25, **kwargs):
        """Set a title"""
        self.ax.set_title(title, pad=pad, **kwargs)

def get_ranges(benchmark):
    """Get the minimum and maximum values for the benchmark and a budget."""
    min_max_per_variable = benchmark.describe().T[['min', 'max']]

    min_acc, max_acc = 1, 0
    min_time, max_time = 1e16, 0
    min_energy, max_energy = 1e16, 0
    min_params, max_params = 1e16, 0

    if min_max_per_variable['min']['accuracy'] < min_acc:
        min_acc = min_max_per_variable['min']['accuracy']
    if min_max_per_variable['max']['accuracy'] > max_acc:
        max_acc = min_max_per_variable['max']['accuracy']

    if min_max_per_variable['min']['time'] < min_time:
        min_time = min_max_per_variable['min']['time']
    if min_max_per_variable['max']['time'] > max_time:
        max_time = min_max_per_variable['max']['time']

    if min_max_per_variable['min']['energy'] < min_energy:
        min_energy = min_max_per_variable['min']['energy']
    if min_max_per_variable['max']['energy'] > max_energy:
        max_energy = min_max_per_variable['max']['energy']

    if min_max_per_variable['min']['params'] < min_params:
        min_params = min_max_per_variable['min']['params']
    if min_max_per_variable['max']['params'] > max_params:
        max_params = min_max_per_variable['max']['params']
    
    return [(min_time, max_time), (min_acc, max_acc), (min_energy, max_energy), (min_params, max_params)]


def get_dataframe(archs, benchmark, budget=108):
    df = pd.DataFrame()
    for i in range(len(archs)):
        matrix, labels = archs[i][0]
        spec = benchmark.get_model_spec(matrix, labels)
        fixed_stats, computed_stats = benchmark.get_metrics_from_spec(spec)
        metrics = {**fixed_stats, **computed_stats[budget][0]}
        df = df.append(metrics, ignore_index=True)

    return df[["final_training_time", "final_validation_accuracy", "energy (kWh)", "trainable_parameters"]]


def get_bend_angles(x, y):
    x = np.array(x)
    y = np.array(y)

    # diff of x to points right
    x_diff = np.diff(x)
    # diff of y to points right
    y_diff = np.diff(y)

    # diff of x to points left
    x_diff_left = np.diff(x[::-1])[::-1]
    # diff of y to points left
    y_diff_left = np.diff(y[::-1])[::-1]

    # angle of line to points right
    angles_right = np.arctan(y_diff, x_diff)
    # angle of line to points left
    angles_left = np.arctan(y_diff_left, x_diff_left)

    # difference of angles
    angles_diff = np.abs(angles_right - angles_left)

    return angles_diff