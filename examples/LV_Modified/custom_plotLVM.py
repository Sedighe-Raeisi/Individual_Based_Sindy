import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
from src.custom_plot import Custom_plot
from src.Dynamical_systems_utils.LV_Modified.LV_M import gt_utils,realparame2gtarray,generate_pdf


save_dir_prefix = "LVM_chk_"

to_plot = [[0, 2], [0, 7], [1, 4],[1,5],[1,7]]
xlabel_list = ["$\\dot{x}$ : $x(1-x)$",
               "$\\dot{x}$ : $-\\frac{xy}{1+hx}$",
               "$\\dot{y}$ : $y$",
               "$\\dot{y}$ : $y^2$",
               "$\\dot{y}$ : $\\frac{xy}{1+hx}$"]

plot_dict = {"legend":True, "est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"black",
             "xlabel_fontsize": 24, "title_fontsize": None,
             "max_y_limit": 40.3,"plot_name":"plot_LVM","pdf_fill":False,"xlabel_list":xlabel_list}

Custom_plot(generate_pdf, pdf_state=True, ground_truth = True, HB_Est = True, FlatB_Est = False,TABLE = False,
            gt_utils=gt_utils, realparame2gtarray=realparame2gtarray, save_dir_prefix=save_dir_prefix,
           fighigth=5, figwidth=12, n_rows=2, n_cols=3,
           scaler="scale", to_plot=to_plot, plot_dict=plot_dict)

