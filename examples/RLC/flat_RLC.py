import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import Flat_HSModel
from src.flat_mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.RLC_Circuit.RLC import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.flat_plot import plt_mcmc
print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 3000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 1
root_path = os.getcwd()
save_dir_prefix = "RLC_chk_"
model = Flat_HSModel

N_param_set = 100
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
                     "q0_info":q0_info, "i0_info":i0_info, "t_info":{}, "noise_info":{"noise_level":0.05}}

mode = "run" # or "run"
print(f"--------------------------- mode = {mode} --------------------------------")
if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict ,
                 NUM_WARMUP=NUM_WARMUP, NUM_CHAINS=NUM_CHAINS, NUM_SAMPLES=NUM_SAMPLES, NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
                 root_path = root_path, save_dir_prefix = save_dir_prefix,
                 program_state = "start", model = model,
                 display_svi = True, mix_data = mix_data, gt_utils = gt_utils)

elif mode=="plot":


    true_params_file_str = f"chk_GT_Data.pkl"
    gt_save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    est_save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith('Flat_'+save_dir_prefix)][0])

    plt_mcmc(gt_save_path,est_save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
                 stop_subplot_n=None, figlength=3,
                 complex_pdf=True, x_range=None, scaler=None)