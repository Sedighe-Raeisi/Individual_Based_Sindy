# Assuming the revised all_plot_combined and Custom_plot_combined functions are in src.AllJobsPlot
from src.AllJobsPlot import all_plot_combined
from src.Dynamical_systems_utils.Lotka_Volterra.Lotka_Volterra import gt_utils, realparame2gtarray, generate_pdf

# --- INPUTS ---
Bh_path = "D:\\BH_HPC_Backup\\SingleBH\\noise25\\chk_LV_dir"

to_plot = [[0, 1], [0, 3], [1, 2], [1, 3]]
xlabel_list = ["$\\dot{x}$ : $x$", "$\\dot{x}$ : $x\\cdot y$", "$\\dot{y}$ : $y$", "$\\dot{y}$ : $x\\cdot y$"]

plot_dict = {"legend": True, "est_color": "blue", "gt_color": "cyan", "flat_color": "green", "pdf_color": "pink",
             "xlabel_fontsize": 21, "title_fontsize": None,
             "max_y_limit": 40.3, "plot_name": "LV_allJobs", "pdf_fill": False, "xlabel_list": xlabel_list}

all_plot_dict = {"Bh_path": Bh_path,
                 "generate_pdf": generate_pdf, "gt_utils": gt_utils, "realparame2gtarray": realparame2gtarray,
                 "flat_save_path": None, "scaler": "scaler", "to_plot": to_plot,
                 "plot_dict": plot_dict, "n_rows": 1,
                 "n_cols": 4, "xlim": None,
                 "fighigth": 3, "figwidth": 12,
                 "pdf_state": True  # Explicitly setting pdf_state
                 }

# --- EXECUTION ---
all_plot_combined(all_plot_dict)