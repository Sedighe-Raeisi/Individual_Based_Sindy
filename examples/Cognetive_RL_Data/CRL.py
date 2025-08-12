import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import MultiTargetMultiEquation_HSModel
from src.mcmc_utils import run_mcmc
from src.Dynamical_systems_utils.Run_from_data.Data_utils import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.plot import plt_mcmc
print("---------------------- parameter defining ------------------------")

NUM_WARMUP = 1000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 5
root_path = os.getcwd()
save_dir_prefix = "CRL_chk_"

model = MultiTargetMultiEquation_HSModel
data_path = "C:\\Users\\s\Downloads\\OsnabrukPostdocProject\\projects\\BH\\physical_system_v6\\src\\Dynamical_systems_utils\\Run_from_data\\data.pkl"
system_param_dict = {"data_path":data_path} # Updated parameter dictionary
mode = "plot" #"plot" or "run"
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