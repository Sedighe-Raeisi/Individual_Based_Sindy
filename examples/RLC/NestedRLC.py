import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import MultiTargetMultiEquation_HSModel
from src.mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.Nested_RLC import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.plot import plt_mcmc
from src.device import device_check
device = device_check()
device.check()

print("---------------------- parameter defining ------------------------")

N_param_set = 100
NUM_WARMUP = 3
NUM_CHAINS = 2
NUM_SAMPLES = 4
NUM_BATCH_SAMPLES = 5
root_path = os.getcwd()
save_dir_prefix = "RLC_chk_"

# Define parameters for RLC circuit
L_info = { "L_mean": 1.0, "L_std": 0.2,"L_N":20}
R_info = {"R_V":1.,"R_mean": 1.0, "R_std":0.01}
C_info = {"C_mean": 0.5, "C_std": 0.1,"C_N":15}
V_in_info = {"V_in_mean": 1.0,"V_in_std":0.03,"V_in_N":1}
q0_info = {"q0_V": 0.0}
i0_info = {"i0_V": 0.0}

model = MultiTargetMultiEquation_HSModel

system_param_dict = {"N_param_set":N_param_set,"L_info":L_info, "R_info":R_info, "C_info":C_info, "V_in_info":V_in_info,
                     "q0_info":q0_info, "i0_info":i0_info, "t_info":{}, "noise_info":{}} # Updated parameter dictionary
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
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
                 stop_subplot_n=None, figlength=3,
                 complex_pdf=True, x_range=None, scaler=None)