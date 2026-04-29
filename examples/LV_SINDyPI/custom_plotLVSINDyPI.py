import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
from src.custom_plot import Custom_plot
from src.Dynamical_systems_utils.LV_SINDyPI.LVM_SINDyPI import gt_utils,realparame2gtarray,generate_pdf


save_dir_prefix = "LVM_SINDyPI_"

to_plot = [ [0,1], [0,2],[0,3],[0,4],[0,5],[0,8],
            [1,6],[1,7],[1,8],[1,9],[1,10],[1,11]]
xlabel_list = [
# "$0$ : $Const.$",
                "$0$ : $x$",
                "$0$ : $x^2$",
                "$0$ : $x^3$",
                "$0$ : $\\dot{x}$",
                "$0$ : $\\dot{x}x$",
                "$0$ : $xy$",

                "$0$ : $y$",
                "$0$ : $y^2$",
                "$0$ : $xy$",
                "$0$ : $x y^2$",
                "$0$ : $\\dot{y}$",
                "$0$ : $x\\dot{y}$"
               ]

plot_dict = {"legend":False, "est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"black",
             "xlabel_fontsize": 24, "title_fontsize": None,
             "max_y_limit": 40.3,"plot_name":"plot_LVMSINDyPI","pdf_fill":False,"xlabel_list":xlabel_list}

Custom_plot(generate_pdf, pdf_state=True, ground_truth = True, HB_Est = True, FlatB_Est = False,TABLE = False,
            gt_utils=gt_utils, realparame2gtarray=realparame2gtarray, save_dir_prefix=save_dir_prefix,
           fighigth=5, figwidth=12, n_rows=2, n_cols=6,
           scaler="z-score", to_plot=to_plot, plot_dict=plot_dict)

