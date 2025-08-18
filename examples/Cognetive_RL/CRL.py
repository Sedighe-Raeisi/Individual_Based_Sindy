import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import MultiTargetMultiEquation_HSModel
from src.mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.Cognetive_RL.CRL_Data_Gen import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.plot import plt_mcmc, row_result
import numpy as np
print("---------------------- parameter defining ------------------------")

N_param_set = 100
NUM_WARMUP = 1000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 5
root_path = os.getcwd()
save_dir_prefix = "CRL_chk_"

# Define parameters for RLC circuit
Alpha_info = { "Alpha_mean": .3, "Alpha_std": 0.02}
ForgetRate_info = {"ForgetRate_mean": .4, "ForgetRate_std":0.01}
Session_info = {"n_trials_per_session":100, "n_sessions":2}

model = MultiTargetMultiEquation_HSModel

system_param_dict = {"N_param_set":N_param_set,"Alpha_info":Alpha_info, "ForgetRate_info":ForgetRate_info, 'Session_info': Session_info} # Updated parameter dictionary
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
    to_plot = [[0,1],[0,2]]
    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    plot_dict = {"est_color": "blue", "gt_color": "green", "legend": None, "xlabel_fontsize": 8, "title_fontsize": None}
    row_result(save_path, gt_utils, realparame2gtarray, true_params_file_str,
               fighigth=4, figwidth=18,
               n_rows=1, n_cols=2,
               scaler=None, to_plot=to_plot, plot_dict=plot_dict)

elif mode=="plot":

    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
                 stop_subplot_n=None, figlength=3,
                 complex_pdf=True, x_range=None, scaler=None)