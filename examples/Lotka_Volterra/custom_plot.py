import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
from src.custom_plot import Custom_plot
from src.Dynamical_systems_utils.Lotka_Voltera import gt_utils,realparame2gtarray


save_dir_prefix = "LV_chk_"

to_plot = [[0,1],[0,3],[1,2],[1,3]]

plot_dict = {"est_color": "blue", "gt_color": "green", "flat_color":"pink",
             "legend": None, "xlabel_fontsize": 6, "title_fontsize": None,
             "max_y_limit": 10}

Custom_plot(ground_truth = True, HB_Est = True, FlatB_Est = True,TABLE = True,
            gt_utils=gt_utils, realparame2gtarray=realparame2gtarray, save_dir_prefix=save_dir_prefix,
           fighigth=3, figwidth=12, n_rows=2, n_cols=2,
           scaler="scale", to_plot=to_plot, plot_dict=plot_dict)

