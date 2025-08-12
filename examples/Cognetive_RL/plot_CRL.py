from src.plot import plot_check_stationary
from src.Dynamical_systems_utils.Cognetive_RL.CRL_Data_Gen import gt_utils
import os
root_path = os.getcwd()
save_dir_prefix = "CRL_chk_"
file_name = [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0]
chk_path = os.path.join(root_path,file_name)

plot_check_stationary(chk_path,gt_utils)