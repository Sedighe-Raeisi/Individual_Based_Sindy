import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
import os.path
from src.model import Flat_HSModel
from src.flat_mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.DampedForced_HO import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.flat_plot import plt_mcmc


print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 1000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 1000
root_path = os.getcwd()
save_dir_prefix = "DFHO_chk_"
model = Flat_HSModel # The model should be general enough for different systems

N_param_set = 100
m_info = {"m_mean":1.0, "m_std":0.2}
k_info = {"k_mean":2.0, "k_std":0.5}
c_info = {"c_mean":0.5, "c_std":0.1}
F0_info = {"F0_mean":0.5, "F0_std":0.1, "F_zero_portion":0.3}
omega_info = {"omega_V":1.5}

system_param_dict = {"N_param_set":N_param_set, "m_info":m_info, "k_info":k_info, "c_info":c_info, "F0_info":F0_info, "omega_info":omega_info,
                     "x0_info":{}, "v0_info":{}, "t_info":{}, "noise_info":{}}

mode = "run" #"plot" or "run"
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