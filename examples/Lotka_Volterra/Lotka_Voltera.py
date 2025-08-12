import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
# Assuming src.model and src.mcmc_utils are still relevant and compatible
# with the new data structure (X_all, Y_all, real_params from mix_data)
from src.model import MultiTargetMultiEquation_HSModel # Still using this model
from src.mcmc_utils import run_mcmc # Still using this utility
from src.plot import plt_mcmc
# from src.dynamical_systems.FitzHughNagumo import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.Dynamical_systems_utils.Lotka_Voltera import mix_data_lotka_volterra, gt_utils, realparame2gtarray, generate_pdf # Import LotkaVolterra functions
import pickle
from src.device import device_check
dv_obj = device_check()
dv_obj.check()
print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 5
NUM_CHAINS = 4
NUM_SAMPLES = 5
NUM_BATCH_SAMPLES = 5
root_path = os.getcwd()
# root_path = os.path.join(root_path,"examples")
# root_path = "C:\\Users\\s\\Downloads\\OsnabrukPostdocProject\\projects\\BH\\physical_system_v4\\examples"
save_dir_prefix = "LV_chk_" # Changed prefix for Lotka-Volterra checkpoints
model = MultiTargetMultiEquation_HSModel # The model should be general enough for different systems

# --- Lotka-Volterra Parameters ---
# Example usage for Lotka-Volterra:
N_param_set = 100
alpha_info = {"alpha_N": 5, "alpha_mean": 1.0, "alpha_std": 0.02}
beta_info = {"beta_N": 5, "beta_mean": 0.1, "beta_std": 0.02}
gamma_info = {"gamma_N": 5, "gamma_mean": 1.5, "gamma_std": 0.03}
delta_info = {"delta_N": 5, "delta_mean": 0.75, "delta_std": 0.03}



# Construct the system_param_dict for Lotka-Volterra
system_param_dict = {"N_param_set":N_param_set,
    "alpha_info": alpha_info,
    "beta_info": beta_info,
    "gamma_info": gamma_info,
    "delta_info": delta_info,
    "x0_info": {}, # Pass initial conditions
    "y0_info": {}, # Pass initial conditions
    "t_info":{},
    "noise_info": {}
}
mode = "run"#"run" # or "plot"
print(f"--------------------------- mode = {mode} --------------------------------")
if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict ,
                 NUM_WARMUP=NUM_WARMUP, NUM_CHAINS=NUM_CHAINS, NUM_SAMPLES=NUM_SAMPLES, NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
                 root_path = root_path, save_dir_prefix = save_dir_prefix,
                 program_state = "start", model = model,
                 display_svi = True, mix_data = mix_data_lotka_volterra, gt_utils = gt_utils,scaler="scale") # Use LotkaVolterra mix_data
elif mode=="plot":

    true_params_file_str = f"chk_GT_Data.pkl" # Updated filename
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    print(save_path)

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
             stop_subplot_n=None, figlength=3,
             complex_pdf=False, x_range=None, scaler="scale")