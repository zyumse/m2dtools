import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

def set_science_style():
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        "lines.markersize": 4,
        "lines.linewidth": 1.5,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans"
    })
    palette = sns.color_palette("muted", n_colors=5)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)