import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
import os.path
import pickle

from src.model import MultiTargetMultiEquation_HSModel
from src.mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.DampedForced_HO import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.plot import plt_mcmc, row_result
import numpy as np


print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 1000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 5
root_path = "." # os.getcwd()
save_dir_prefix = "DFHO_chk_"
model = MultiTargetMultiEquation_HSModel

N_param_set = 100
m_info = {"m_mean":1.0, "m_std":0.2}
k_info = {"k_mean":2.0, "k_std":0.5}
c_info = {"c_mean":0.5, "c_std":0.1}
F0_info = {"F0_mean":0.5, "F0_std":0.1, "F_zero_portion":0.3}
omega_info = {"omega_V":1.5}

system_param_dict = {"N_param_set":N_param_set, "m_info":m_info, "k_info":k_info, "c_info":c_info, "F0_info":F0_info, "omega_info":omega_info,
                     "x0_info":{}, "v0_info":{}, "t_info":{}, "noise_info":{}}

mode = "row_plot" #"plot" or "run" or "row_plot"
print(f"--------------------------- mode = {mode} --------------------------------")
if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict ,
                 NUM_WARMUP=NUM_WARMUP, NUM_CHAINS=NUM_CHAINS, NUM_SAMPLES=NUM_SAMPLES, NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
                 root_path = root_path, save_dir_prefix = save_dir_prefix,
                 program_state = "start", model = model,
                 display_svi = True, mix_data = mix_data, gt_utils = gt_utils)

elif mode == "row_plot":
    to_plot = [[1,1],[1,2],[1,3]]

    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    plot_dict = {"est_color": "blue", "gt_color": "green", "legend": None, "xlabel_fontsize": 8, "title_fontsize": None}
    row_result(save_path, gt_utils, realparame2gtarray, true_params_file_str,
               fighigth=4, figwidth=18,
               n_rows=1, n_cols=3,
               scaler=None, to_plot=to_plot, plot_dict=plot_dict)

elif mode=="plot":

    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    # plt_mcmc(save_path,gt_utils,realparame2gtarray, generate_pdf, true_params_file_str,4,complex_pdf=True)

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
            stop_subplot_n=None, figlength=3,
            complex_pdf=True, x_range=None, scaler=None)