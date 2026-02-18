from src.plot import plot_all_traj
from src.Dynamical_systems_utils.Damped_Oscillator.DampedForced_HO import gt_utils,realparame2gtarray, generate_pdf
from src.plot import plot_check_stationary
import os
root_path = os.getcwd()

save_dir_prefix = "HO_chk_" # Changed prefix for Lotka-Volterra checkpoints

true_params_file_str = f"chk_GT_Data.pkl" # Updated filename
save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
print(save_path)
plot_all_traj(save_path,gt_utils,realparame2gtarray, true_params_file_str,4)
