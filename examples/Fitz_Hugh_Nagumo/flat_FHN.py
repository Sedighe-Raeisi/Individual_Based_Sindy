import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import Flat_HSModel
from src.flat_mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.FitzHugh_Nagumo.FitzHughNagumo import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.flat_plot import plt_mcmc
print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 100
NUM_CHAINS = 1
NUM_SAMPLES = 500
NUM_BATCH_SAMPLES = 1
root_path = os.getcwd()
save_dir_prefix = "FHN_chk_"
model = Flat_HSModel

# --- FitzHugh-Nagumo Parameters ---
N_param_set = 10
a_info = {"a_N":10, "a_mean": 0.5, "a_std":0.1}
b0_info = {"b0_N":10, "b0_mean": 2.0, "b0_std":0.2}
b1_info = {"b1_mean": 2.0, "b1_std":0.2}
I_info = {"I_mean": 1.5, "I_std":0.1}

# Initial conditions (v0, w0) can be fixed or sampled from a distribution
v0_info = {"v0_V": 0.5} # initial value for v
w0_info = {"w0_V": 0.5} # initial value for w

# Time info and noise info can remain as before, or adjusted as needed
t_info = {"t_start": 0, "t_end": 10, "dt": 0.01} # time span
noise_info = {"noise_std": 0.01} # noise level

# Construct the system_param_dict for FitzHugh-Nagumo
system_param_dict = {"N_param_set":N_param_set,
    "a_info": a_info,
    "b0_info": b0_info,
    "b1_info": b1_info,
    "I_info": I_info,
    "v0_info": v0_info, # Pass initial conditions
    "w0_info": w0_info, # Pass initial conditions
    "t_info": t_info,
    "noise_info":{"noise_level":0.25}
}
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