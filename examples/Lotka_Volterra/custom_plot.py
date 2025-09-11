import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
from src.custom_plot import Custom_plot
from src.Dynamical_systems_utils.Lotka_Voltera import gt_utils,realparame2gtarray,generate_pdf


save_dir_prefix = "LV_chk_"

to_plot = [[0,1],[0,3],[1,2],[1,3]]
xlabel_list = ["$\\dot{x}$:$x$","$\\dot{x}$:$x\\cdot y$","$\\dot{y}$:$y$","$\\dot{y}$:$x\\cdot y$"]

plot_dict = {"legend":False,"est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"grey",
             "xlabel_fontsize": 24, "title_fontsize": None,
             "max_y_limit": 40.3,"save_name":"custom_with_pdf","pdf_fill":False,"xlabel_list":xlabel_list}

Custom_plot(generate_pdf, pdf_state=True, ground_truth = True, HB_Est = True, FlatB_Est = True,TABLE = False,
            gt_utils=gt_utils, realparame2gtarray=realparame2gtarray, save_dir_prefix=save_dir_prefix,
           fighigth=4, figwidth=12, n_rows=1, n_cols=4,
           scaler="scale", to_plot=to_plot, plot_dict=plot_dict)

