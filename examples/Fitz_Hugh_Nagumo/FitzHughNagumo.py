import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
import os.path
from src.model import MultiTargetMultiEquation_HSModel,Double_HSModel
from src.mcmc_utils import run_mcmc
from src.plot import plt_mcmc
from src.Dynamical_systems_utils.FitzHugh_Nagumo.FitzHughNagumo import mix_data,gt_utils,realparame2gtarray, generate_pdf
import pickle
import os
from src.plot import plt_mcmc, row_result
from src.overfit_report import path2idata, _generate_overfitting_report
import numpy as np
np.random.seed(42)

print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 2000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 1
root_path = os.getcwd()
save_dir_prefix = "FHN_chk_"
model = MultiTargetMultiEquation_HSModel

N_param_set = 100
a_info = {"a_N":10, "a_mean": 0.5, "a_std":0.1}
b0_info = {"b0_N":10, "b0_mean": 2.0, "b0_std":0.2}
b1_info = {"b1_mean": 2.0, "b1_std":0.1}
I_info = {"I_mean": 1.5, "I_std":0.1}

# Initial conditions (v0, w0) can be fixed or sampled from a distribution
# For initial testing, we fix them.
v0_info = {"v0_V": 0.5} # initial value for v
w0_info = {"w0_V": 0.5} # initial value for w

# Time info and noise info can remain as before, or adjusted as needed
t_info = {"t_start": 0, "t_end": 10, "dt": 0.01}
#noise_info = {"noise_std": 0.01}

# Construct the system_param_dict for FitzHugh-Nagumo
system_param_dict = {"N_param_set":N_param_set,
    "a_info": a_info,
    "b0_info": b0_info,
    "b1_info": b1_info,
    "I_info": I_info,
    "v0_info": v0_info, # Pass initial conditions
    "w0_info": w0_info, # Pass initial conditions
    "t_info": t_info,
    "noise_info":{"noise_level":0.05}
}
mode = "run" # or "run" or "plot" or "row_plot"
print(f"--------------------------- mode = {mode} --------------------------------")
if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict ,
                 NUM_WARMUP=NUM_WARMUP, NUM_CHAINS=NUM_CHAINS, NUM_SAMPLES=NUM_SAMPLES, NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
                 root_path = root_path, save_dir_prefix = save_dir_prefix,
                 program_state = "start", model = model,
                 display_svi = True, mix_data = mix_data, gt_utils = gt_utils)
                 
    print("**************** Evaluation of model based on LOO metric for overfit *******************")
    hb_save_dir_prefix = "FHN_chk_"
    root_path = os.getcwd()
    idata = path2idata(root_path,hb_save_dir_prefix,scaler=None)
    _generate_overfitting_report(idata,model)            
                 
elif mode == "row_plot":
    to_plot = [[1,0],[1,1]]


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