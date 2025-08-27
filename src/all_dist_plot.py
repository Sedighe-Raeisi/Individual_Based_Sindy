
################## row plot ######################

import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from scipy.stats import gaussian_kde


from src.Dynamical_systems_utils.Cognetive_RL.CRL_Data_Gen import gt_utils as crl_gt_utils
from src.Dynamical_systems_utils.Cognetive_RL.CRL_Data_Gen import realparame2gtarray as crl_realparame2gtarray
from src.Dynamical_systems_utils.Cognetive_RL.CRL_Data_Gen import generate_pdf as crl_generate_pdf

from src.Dynamical_systems_utils.DampedForced_HO import gt_utils as ho_gt_utils
from src.Dynamical_systems_utils.DampedForced_HO import realparame2gtarray as ho_realparame2gtarray
from src.Dynamical_systems_utils.DampedForced_HO import generate_pdf as ho_generate_pdf

from src.Dynamical_systems_utils.FitzHughNagumo import gt_utils as fhn_gt_utils
from src.Dynamical_systems_utils.FitzHughNagumo import realparame2gtarray as fhn_realparame2gtarray
from src.Dynamical_systems_utils.FitzHughNagumo import generate_pdf as fhn_generate_pdf

from src.Dynamical_systems_utils.Lotka_Voltera import gt_utils as lv_gt_utils
from src.Dynamical_systems_utils.Lotka_Voltera import realparame2gtarray as lv_realparame2gtarray
from src.Dynamical_systems_utils.Lotka_Voltera import generate_pdf as lv_generate_pdf


CRL_info = {"system_name":"Cognitive Reinforcement learning","pdf_func":crl_generate_pdf, "gt_utils":crl_gt_utils,"realparame2gtarray":crl_realparame2gtarray,
             "data_path":"examples\\Cognetive_RL","save_dir_prefix":"CRL_chk_",
            "to_plot": [[0,1],[0,2]], "scale" : None}

HO_info = {"system_name":"Damped Forced Harmonic Oscillator","pdf_func":ho_generate_pdf, "gt_utils":ho_gt_utils,"realparame2gtarray":ho_realparame2gtarray,
             "data_path":"examples\\Damped_Forced_Harmonic_Oscillator","save_dir_prefix":"DFHO_chk_",
           "to_plot" : [[1,1],[1,2],[1,3]], "scale" : None}

FHN_info = {"system_name":"Fitz Hugh Nagumo","pdf_func":fhn_generate_pdf, "gt_utils":fhn_gt_utils,"realparame2gtarray":fhn_realparame2gtarray,
             "data_path":"examples\\Fitz_Hugh_Nagumo","save_dir_prefix":"FHN_chk_",
            "to_plot" : [[1,0],[1,1]], "scale" : None}

LV_info = {"system_name":"Lotka Volterra","pdf_func":lv_generate_pdf, "gt_utils":lv_gt_utils,"realparame2gtarray":lv_realparame2gtarray,
             "data_path":"examples\\Lotka_Volterra","save_dir_prefix":"LV_chk_",
           "to_plot" : [[0,1],[0,3],[1,2],[1,3]], "scale" : "scale"}

systems_info= [CRL_info,HO_info,FHN_info,LV_info]
#######################################################################################

def one_subplot(info,fig,plot_dict,
                ground_truth = True, HB_Est = True, FlatB_Est = True,pdf_state = True,
                n_rows=None, n_cols=None,start_row=0):

    root_path = os.path.join(os.path.dirname(os.getcwd()),info['data_path'])

    save_dir_prefix = info["save_dir_prefix"]

    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])

    legend_True = True if plot_dict is not None and plot_dict.get('legend') else None
    xlabel_fontsize = plot_dict.get('xlabel_fontsize', None)
    title_fontsize = plot_dict.get('title_fontsize', None)
    est_color = plot_dict.get('est_color', 'blue')
    gt_color = plot_dict.get('gt_color', 'pink')
    flat_color = plot_dict.get('flat_color', 'grey')
    pdf_color = plot_dict.get("pdf_color", "red")
    max_y_limit = plot_dict.get("max_y_limit", 0.98)
    plot_name = plot_dict.get("plot_name", 'custom_plot.jpg')
    ############################# PDF #####################################
    generate_pdf = info["pdf_func"]
    if pdf_state:
        pdf_arr = generate_pdf(save_path)

    ############################## Load GT Data ###########################
    true_params_file_str = f"chk_GT_Data.pkl"
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)
    realparame2gtarray = info["realparame2gtarray"]
    gt_coef_array = realparame2gtarray(true_params)
    gt_utils = info["gt_utils"]
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    N_Indv, N_Coef, N_Obs = X_data.shape
    N_Indv, N_Eqs, N_Obs = Y_data.shape
    ########################## Load EST data #############################
    if info["scale"] is None:
        npz_sample_path = os.path.join(save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        est_coef = loaded_samples['coef']
        est_coef_array = np.array(est_coef)
    else:
        with open(os.path.join(save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            print(f" start loading samples from {os.path.join(save_path, f'revert_mcmc_samples.pkl')}")
            mcmc_coef_results = pickle.load(f)
            est_coef_array = np.array(mcmc_coef_results)

    print(f"est_coef_array.shape = {est_coef_array.shape}")
    est_coef_mean_arr = np.mean(est_coef_array, axis=0)
    ########################## Load Flat EST data #############################
    flat_save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if
                                              file.startswith('Flat_' + save_dir_prefix)][0])
    if info["scale"] is None:

        npz_sample_path = os.path.join(flat_save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        flat_est_coef = loaded_samples['coef']
        flat_est_coef_array = np.array(flat_est_coef)
    else:
        with open(os.path.join(flat_save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            flat_mcmc_coef_results = pickle.load(f)
            flat_est_coef_array = np.array(flat_mcmc_coef_results)

    print(f"est_coef_array.shape = {flat_est_coef_array.shape}")
    flat_est_coef_mean_arr = np.mean(flat_est_coef_array, axis=0)

    ################### Prepare plot data and table data ##################

    mpl.rcParams['xtick.labelsize'] = 6
    missed_coef_data = []



    ploted_eqs = []
    plot_counter = 0
    used_axes = []

    # Loop to create and populate the subplots for the specified indices
    set_indx0 = []
    set_indx1 = []
    row_idx = -1
    col_idx = -1
    for index in info["to_plot"]:
        if index[0] not in set_indx0:
            set_indx0 +=[index[0]]
            row_idx +=1
            col_idx = -1
            set_indx1 = []
        if index[1] not in set_indx1:
            set_indx1 +=[index[1]]
            col_idx +=1

        row_plot_idx = plot_counter // n_cols
        col = plot_counter % n_cols

        # Add a row title if it's a new row
        if index[0] not in ploted_eqs:
            # title_ax = fig.add_subplot(gs[1 * row_plot_idx, 0:n_cols])
            # title_ax.set_title(f'{eqs[index[0]]}', fontsize=12, fontweight='bold')
            # title_ax.axis('off')
            ploted_eqs.append(index[0])

        # Plot in the dedicated subplot row
        axi = plt.subplot2grid((n_rows, n_cols), (start_row+row_idx, col_idx), colspan=1, fig=fig)
        # used_axes.append(axi)
        y_lim = 0
        try:
            if HB_Est:
                print("HB add")
                sns.kdeplot(est_coef_mean_arr[:, index[1], index[0]], ax=axi, fill=False, color=est_color, alpha=.7,
                            warn_singular=False, linewidth=3, ls='--')
                data = est_coef_mean_arr[:, index[1], index[0]]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim, peak_y)

            if ground_truth:
                print("gt add")
                if true_params:
                    sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=False, color=gt_color, alpha=.9,
                                warn_singular=False, linewidth=1)

                    data = gt_coef_array[index[0], index[1], :]
                    kernel = gaussian_kde(data)
                    x_values = np.linspace(min(data), max(data), 100)
                    y_values = kernel(x_values)
                    kernel = gaussian_kde(data)
                    peak_y = np.max(y_values)
                    y_lim = max(y_lim, peak_y)
            if pdf_state:
                print("pdf add")
                print(pdf_arr.shape)
                sns.kdeplot(pdf_arr[index[0], index[1], :], ax=axi, fill=False, color=pdf_color, alpha=.8,
                            warn_singular=False, linewidth=2)
                # data = pdf_arr[index[0], index[1],:]
                # kernel = gaussian_kde(data)
                # x_values = np.linspace(min(data), max(data), 100)
                # y_values = kernel(x_values)
                # kernel = gaussian_kde(data)
                # peak_y = np.max(y_values)
                # y_lim = max(y_lim, peak_y)

        except:
            if HB_Est:
                print("HB add")
                sns.kdeplot(est_coef_mean_arr[:, index[1], index[0]], ax=axi, fill=False, color=est_color, alpha=.7,
                            warn_singular=False, linewidth=3, ls='--')
                data = est_coef_mean_arr[:, index[1], index[0]]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim, peak_y)

            if ground_truth:
                print("gt add")
                if true_params:
                    sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=False, color=gt_color, alpha=.9,
                                warn_singular=False, linewidth=1)
                    data = gt_coef_array[index[0], index[1], :]
                    kernel = gaussian_kde(data)
                    x_values = np.linspace(min(data), max(data), 100)
                    y_values = kernel(x_values)
                    kernel = gaussian_kde(data)
                    peak_y = np.max(y_values)
                    y_lim = max(y_lim, peak_y)

            if pdf_state:
                print("pdf add")
                sns.kdeplot(pdf_arr[index[0], index[1], :], ax=axi, fill=False, color=pdf_color, alpha=.9,
                            warn_singular=False, linewidth=4)
                data = pdf_arr[index[0], index[1], :]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim, peak_y)

        if FlatB_Est:
            print("flat add")
            print("flat shape", flat_est_coef_array.shape)
            sns.kdeplot(flat_est_coef_array[:, index[1], index[0]], ax=axi, fill=False, color=flat_color, alpha=.7,
                        warn_singular=False, linewidth=2, ls='-')
            # max_y_limit = .98  # A good starting value, adjust as needed
        axi.set_ylim(0, y_lim * 1.2)

        axi.set_xlabel(f"{info["system_name"]}\n{eqs[index[0]].split("=")[0]} : {coef_names[index[1]]}", fontsize=xlabel_fontsize)
        axi.set_ylabel(" ", fontsize=xlabel_fontsize)
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
        axi.set_yticks(np.array([0]))
        axi.set_yticklabels([""], fontsize=12, rotation=90)
        axi.yaxis.set_tick_params(length=0)
        axi.spines['left'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        if legend_True:
            axi.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=1, fancybox=True, shadow=True, fontsize=8)

        plot_counter += 1


########################################################################################################################
def All_plot(systems_info,
                ground_truth = True, HB_Est = True, FlatB_Est = True,pdf_state = True,
                # gt_utils=None, realparame2gtarray=None, save_dir_prefix=None,
               fighigth=3, figwidth=12, n_rows=None, n_cols=None,
               scaler=None, to_plot=None, plot_dict=dict()):

    plot_name = plot_dict.get("plot_name", 'custom_plot.jpg')
    mpl.rcParams['xtick.labelsize'] = 6

    fig = plt.figure(layout="constrained",figsize=(figwidth,fighigth))
    row_plot_idx = 0
    i=0
    rowspan_list = [1,1,1,2]
    label = ["a","b","c","d"]
    for system_info in systems_info:
        print("row_plot_idx = ", row_plot_idx)
        one_subplot(info=system_info, fig=fig, plot_dict=plot_dict,
                    ground_truth=ground_truth, HB_Est=HB_Est, FlatB_Est=FlatB_Est, pdf_state=pdf_state,
                    n_rows=n_rows, n_cols=n_cols,start_row=row_plot_idx)
        used_row = len(set([index[0] for index in system_info['to_plot']]))
        row_plot_idx+=used_row
        # ax = plt.subplot2grid((n_rows, n_cols), (row_plot_idx, 0), colspan=n_cols,rowspan=rowspan_list[i], fig=fig)
        # ax.text(0., 0.5, label[i], va="center", ha="center")
        # i+=1
        # ax.tick_params(labelbottom=False, labelleft=False)

    plt.subplots_adjust(
        left=0.05,  # Left margin
        right=0.9,  # Right margin (pulls subplots left)
        bottom=0.2,  # Bottom margin (pushes subplots up)
        wspace=0.6  # Width space between subplots
    )
    plt.savefig(os.path.join(os.getcwd(), plot_name))
    plt.show()
    print("plot run finished")

##############################################################################
plot_dict = {"est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"grey",
             "legend": None, "xlabel_fontsize": 7, "title_fontsize": None,
             "max_y_limit": 40.3,"plot_name":"All_with_pdf.jpg"}
All_plot(systems_info,
                ground_truth = True, HB_Est = True, FlatB_Est = True,pdf_state = True,
                # gt_utils=None, realparame2gtarray=None, save_dir_prefix=None,
               fighigth=10, figwidth=12, n_rows=5, n_cols=3,
               scaler=None, to_plot=None, plot_dict=plot_dict)