import os
import pickle
import numpy as np

# Set environment before other imports
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import MultiTargetMultiEquation_HSModel
from src.mcmc_utils import run_mcmc
from src.plot import plt_mcmc, row_result
from src.device import device_check

# --- Import updated LV_M functions ---
# Ensure these are the versions that handle the 11-term library
from src.Dynamical_systems_utils.LV_Modified.LV_M import (
    mix_data_LV_M,  # Using the revised mix function
    gt_utils,
    realparame2gtarray,
    generate_pdf  # The edited version from previous step
)

dv_obj = device_check()
dv_obj.check()

print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 3000
NUM_CHAINS = 6
NUM_SAMPLES = 2000
NUM_BATCH_SAMPLES = 10
root_path = os.getcwd()
save_dir_prefix = "LVM_chk_"  # Updated prefix for Modified LV
model = MultiTargetMultiEquation_HSModel

print(f"NUM_WARMUP = {NUM_WARMUP} , NUM_SAMPLES = {NUM_SAMPLES} , NUM_CHAINS = {NUM_CHAINS}")
# --- Expanded Parameter Sets for LV_M ---
N_param_set = 100


# Helper to create info dicts quickly
def mk_info(mean, std, N=5):
    return {"mean": mean, "std": std, "N": N, "V": mean}


# --- Parameter defining ---
N_param_set = 100

# Standard & Modified LV Parameters
alpha_info = {"alpha_N": None, "alpha_mean": 1.0, "alpha_std": 0.05, "alpha_V": None}
beta_info = {"beta_N": None, "beta_mean": 1.7, "beta_std": 0.01, "beta_V": None}
h_info = {"h_N": None, "h_mean": None, "h_std": None, "h_V": 0.4}
epsilon_info = {"epsilon_N": None, "epsilon_mean": None, "epsilon_std": None, "epsilon_V": 0.1}
m_info = {"m_N": None, "m_mean": 0.4, "m_std": 0.01, "m_V": None}
H_info = {"H_N": None, "H_mean": 0.075, "H_std": 0.025, "H_V": None}

# Placeholder Deltas (Required for the 11-term library)
delta1_info = {"delta1_N": None, "delta1_mean": None, "delta1_std": None, "delta1_V": 0.0}
delta2_info = {"delta2_N": None, "delta2_mean": None, "delta2_std": None, "delta2_V": 0.0}
delta3_info = {"delta3_N": None, "delta3_mean": None, "delta3_std": None, "delta3_V": 0.0}
delta4_info = {"delta4_N": None, "delta4_mean": None, "delta4_std": None, "delta4_V": 0.0}

# Construct the system_param_dict for LV_M
system_param_dict = {
    "N_param_set": N_param_set,
    "alpha_info": alpha_info,
    "beta_info": beta_info,
    "h_info": h_info,
    "epsilon_info": epsilon_info,
    "m_info": m_info,
    "H_info": H_info,
    "delta1_info": delta1_info,
    "delta2_info": delta2_info,
    "delta3_info": delta3_info,
    "delta4_info": delta4_info,

    # Simulation and Initial Conditions
    "x0_info": {"x0_V": 10},
    "y0_info": {"y0_V": 10},
    "t_info": {"t_start": 0, "t_end": 20, "N_t": 1000},
    "noise_info": {"noise_level": 0.01}
}
mode = "run"
print(f"--------------------------- mode = {mode} --------------------------------")

if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict,
             NUM_WARMUP=NUM_WARMUP,
             NUM_CHAINS=NUM_CHAINS,
             NUM_SAMPLES=NUM_SAMPLES,
             NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
             root_path=root_path,
             save_dir_prefix=save_dir_prefix,
             program_state="start",
             model=model,
             display_svi=True,
             mix_data=mix_data_LV_M,  # Pointing to the M-version
             gt_utils=gt_utils,
             scaler="scale")

elif mode == "row_plot":
    # indices adapted for the 11-term library
    # e.g., [0, 2] maps Const vs alpha*x(1-x)
    to_plot = [[0, 2], [0, 7], [1, 4],[1,5]]

    true_params_file_str = "chk_GT_Data.pkl"
    # Find the directory based on the prefix
    matching_dirs = [d for d in os.listdir(root_path) if d.startswith(save_dir_prefix) and os.path.isdir(d)]
    if not matching_dirs:
        raise FileNotFoundError(f"No directory found with prefix {save_dir_prefix}")

    save_path = os.path.join(root_path, matching_dirs[0])
    plot_dict = {"est_color": "blue", "gt_color": "green", "legend": None, "xlabel_fontsize": 8, "title_fontsize": None}

    row_result(save_path, gt_utils, realparame2gtarray, true_params_file_str,
               fighigth=4, figwidth=18,
               n_rows=2, n_cols=2,
               scaler="scale", to_plot=to_plot, plot_dict=plot_dict)

elif mode == "plot":
    true_params_file_str = "chk_GT_Data.pkl"
    matching_dirs = [d for d in os.listdir(root_path) if d.startswith(save_dir_prefix) and os.path.isdir(d)]
    save_path = os.path.join(root_path, matching_dirs[0])
    print(f"Plotting from: {save_path}")

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
             stop_subplot_n=None, figlength=3,
             complex_pdf=True, x_range=None, scaler="scale")