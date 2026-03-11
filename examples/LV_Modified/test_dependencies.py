import os
import numpy as np
import pickle
from src.Dynamical_systems_utils.LV_Modified.LV_M import mix_data_LV_M
# --- Environment Setup ---
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
import matplotlib.pyplot as plt
# --- Import Core Model Utilities ---
from src.model import MultiTargetMultiEquation_HSModel
from src.mcmc_utils import run_mcmc
from src.plot import plt_mcmc, row_result
from src.device import device_check

# --- Import YOUR custom LV_M functions ---
# Assuming these are in a file named lv_m_logic.py or within the same script
# From your second block: mix_data_LV_M, gt_utils, realparame2gtarray
# Also need generate_pdf from your original utils
from src.Dynamical_systems_utils.Lotka_Voltera import generate_pdf

# --- Device Initialization ---
dv_obj = device_check()
dv_obj.check()

print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 1000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 1000
root_path = os.getcwd()
save_dir_prefix = "LVM_Modified_chk_"
model_type = MultiTargetMultiEquation_HSModel

# --- LV_M Parameters (Expanded to match your mix_data_LV_M) ---
N_param_set = 100

# Mapping the 6 main parameters + dummy/placeholders for the expanded library
system_param_dict = {
    "N_param_set": N_param_set,
    "alpha_info": {"alpha_V": 1.0, "alpha_mean": 1.0, "alpha_std": 0.05},
    "beta_info": {"beta_V": 2.0, "beta_mean": 2.0, "beta_std": 0.05},
    "h_info": {"h_V": 0.5, "h_mean": 0.5, "h_std": 0.02},
    "epsilon_info": {"epsilon_V": 0.5, "epsilon_mean": 0.5, "epsilon_std": 0.02},
    "m_info": {"m_V": 0.3, "m_mean": 0.3, "m_std": 0.02},
    "H_info": {"H_V": 0.05, "H_mean": 0.05, "H_std": 0.01},
    # Delta parameters for the expanded library terms (h4, h5)
    "delta1_info": {"delta1_mean": 0.0, "delta1_std": 0.01},
    "delta2_info": {"delta2_mean": 0.0, "delta2_std": 0.01},
    "delta3_info": {"delta3_mean": 0.0, "delta3_std": 0.01},
    "delta4_info": {"delta4_mean": 0.0, "delta4_std": 0.01},
    # Simulation settings
    "x0_info": {"x0_V": 0.8},
    "y0_info": {"y0_V": 0.8},
    "t_info": {"t_start": 0, "t_end": 50, "N_t": 1000},
    "noise_info": {"noise_level": 0.1}
}
x_all, y_all , real_param = mix_data_LV_M(system_param_dict)
print(x_all.shape)
print(y_all.shape)
for i in range(x_all.shape[0]):
    plt.plot(x_all[i,1,:],x_all[i,4,:])
plt.show()
# mode = "run"  # "run" | "plot" | "row_plot"
# print(f"--------------------------- mode = {mode} --------------------------------")
#
# if mode == "run":
#     print("----------------------- run mcmc_utils -----------------------")
#     run_mcmc(system_param_dict,
#              NUM_WARMUP=NUM_WARMUP,
#              NUM_CHAINS=NUM_CHAINS,
#              NUM_SAMPLES=NUM_SAMPLES,
#              NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
#              root_path=root_path,
#              save_dir_prefix=save_dir_prefix,
#              program_state="start",
#              model=model_type,
#              display_svi=True,
#              mix_data=mix_data_LV_M,  # Points to your new mix function
#              gt_utils=gt_utils,  # Points to your new gt_utils
#              scaler="scale")
#
# elif mode == "row_plot":
#     # Indexing into the 11 terms defined in your new gt_utils
#     # 2=g2(x), 7=Type II Interaction, 4=Y(h2), 5=Y^2(h3)
#     to_plot = [[0, 2], [0, 7], [1, 4], [1, 7]]
#
#     true_params_file_str = f"chk_GT_Data.pkl"
#     # Find the directory created during "run"
#     matching_dirs = [d for d in os.listdir(root_path) if d.startswith(save_dir_prefix)]
#     if not matching_dirs:
#         raise FileNotFoundError("No checkpoint directory found. Run in 'run' mode first.")
#
#     save_path = os.path.join(root_path, matching_dirs[0])
#     plot_dict = {"est_color": "blue", "gt_color": "green", "legend": None, "xlabel_fontsize": 8, "title_fontsize": None}
#
#     row_result(save_path, gt_utils, realparame2gtarray, true_params_file_str,
#                fighigth=4, figwidth=18,
#                n_rows=2, n_cols=2,
#                scaler="scale", to_plot=to_plot, plot_dict=plot_dict)
#
# elif mode == "plot":
#     true_params_file_str = f"chk_GT_Data.pkl"
#     matching_dirs = [d for d in os.listdir(root_path) if d.startswith(save_dir_prefix)]
#     save_path = os.path.join(root_path, matching_dirs[0])
#
#     plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
#              stop_subplot_n=None, figlength=3,
#              complex_pdf=True, x_range=None, scaler="scale")