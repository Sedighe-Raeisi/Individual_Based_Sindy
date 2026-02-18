import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
import os.path
from src.model import MultiTargetMultiEquation_HSModel,TStudent_BHModel
from src.mcmc_utils import run_mcmc
from src.plot import plt_mcmc
from src.Dynamical_systems_utils.RLC_Circuit.RLC import mix_data,gt_utils,realparame2gtarray, generate_pdf
import pickle
import os
from src.plot import plt_mcmc, row_result
import numpy as np

print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 100
NUM_CHAINS = 1
NUM_SAMPLES = 500
NUM_BATCH_SAMPLES = 1
root_path = os.getcwd()
save_dir_prefix = "RLC_chk_"
model = TStudent_BHModel #MultiTargetMultiEquation_HSModel

N_param_set = 10
# Define parameters for RLC circuit
L_info = { "L_mean": 1.0, "L_std": 0.005,
           "L_2Posrtion":0.5 ,"L_2mean":4.0 , "L_2std":0.005}

R_info = {"R_mean": 1.0, "R_std":0.01} #,
          # "R_2Posrtion":0.5 ,"R_2mean":2.0 , "R_2std":0.01}
C_info = {"C_mean": 0.5, "C_std": 0.01}
V_in_info = {"V_in_mean": 1.0,"V_in_std":0.03,"V_in_N":3}
q0_info = {"q0_V": 0.0}
i0_info = {"i0_V": 0.0}

# Construct the system_param_dict for FitzHugh-Nagumo
system_param_dict = {"N_param_set":N_param_set,"L_info":L_info, "R_info":R_info, "C_info":C_info, "V_in_info":V_in_info,
                     "q0_info":q0_info, "i0_info":i0_info, "t_info":{}, "noise_info":{"noise_level":0.25}}

mode = "run" # or "run" or "plot" or "row_plot"
print(f"--------------------------- mode = {mode} --------------------------------")
if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict ,
                 NUM_WARMUP=NUM_WARMUP, NUM_CHAINS=NUM_CHAINS, NUM_SAMPLES=NUM_SAMPLES, NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
                 root_path = root_path, save_dir_prefix = save_dir_prefix,
                 program_state = "start", model = model,
                 display_svi = True, mix_data = mix_data, gt_utils = gt_utils)
elif mode == "row_plot":
    to_plot = [[1,0],[1,1],[1,2]]


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

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
             stop_subplot_n=None, figlength=3,
             complex_pdf=True, x_range=None, scaler=None)