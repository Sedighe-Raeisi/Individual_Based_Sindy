from src.custom_plot import Custom_plot
from src.Dynamical_systems_utils.FitzHughNagumo import gt_utils,realparame2gtarray,generate_pdf


save_dir_prefix = "FHN_chk_"

to_plot = [[1,0],[1,1],[1,2]]
xlabel_list = ["$\\dot{w}$ : $\\text{Const.}$","$\\dot{w}$ : $v$","$\\dot{w}$ : $w$"]

plot_dict = {"legend":False,"est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"grey",
             "xlabel_fontsize": 24, "title_fontsize": None,
             "max_y_limit": 40.3,"save_name":"custom_with_pdf","pdf_fill":False,"xlabel_list":xlabel_list}

Custom_plot(generate_pdf, pdf_state=True, ground_truth = True, HB_Est = True, FlatB_Est = True,TABLE = False,
            gt_utils=gt_utils, realparame2gtarray=realparame2gtarray, save_dir_prefix=save_dir_prefix,
           fighigth=3, figwidth=12, n_rows=1, n_cols=3,
           scaler=None, to_plot=to_plot, plot_dict=plot_dict)

