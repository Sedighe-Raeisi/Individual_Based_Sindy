import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
import os.path
from src.model import MultiTargetMultiEquation_HSModel,Double_HSModel
from src.mcmc_utils import run_mcmc
from src.plot import plt_mcmc
from src.Dynamical_systems_utils.Lotka_Volterra.Lotka_Volterra import mix_data,gt_utils,realparame2gtarray, generate_pdf
import pickle
import os
from src.plot import plt_mcmc, row_result
from src.overfit_report import path2idata, _generate_overfitting_report
import numpy as np
#np.random.seed(42)


print("---------------------- parameter defining ------------------------")
NUM_WARMUP = 3000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 1
root_path = os.getcwd()
save_dir_prefix = "LV_chk_"
##############################################

model = MultiTargetMultiEquation_HSModel

##############################################

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
                 display_svi = True, mix_data = mix_data, gt_utils = gt_utils,scaler="scale")
                 
    print("**************** Evaluation of model based on LOO metric for overfit *******************")
    hb_save_dir_prefix = "LV_chk_"
    root_path = os.getcwd()
    idata = path2idata(root_path,hb_save_dir_prefix,scaler="scale")
    _generate_overfitting_report(idata,model)
                     
    
elif mode == "row_plot":
    to_plot = [[0, 1], [0, 3], [1, 2], [1, 3]]


    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    plot_dict = {"est_color": "blue", "gt_color": "green", "legend": None, "xlabel_fontsize": 8, "title_fontsize": None}
    row_result(save_path, gt_utils, realparame2gtarray, true_params_file_str,
               fighigth=4, figwidth=18,
               n_rows=1, n_cols=2,
               scaler="scale", to_plot=to_plot, plot_dict=plot_dict)

elif mode=="plot":

    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])

    plt_mcmc(save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
             stop_subplot_n=None, figlength=3,
             complex_pdf=True, x_range=None, scaler="scale")