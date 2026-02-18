import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from src.mcmc_utils import BH_scaler
from src.Dynamical_systems_utils.Damped_Oscillator.DampedForced_HO import gt_utils,realparame2gtarray,generate_pdf
from src.traj_validating import plot_predictions_vs_truth
prefix = "HO_chk"
scaler = None
save_dir = os.getcwd()

save_name = "val_traj_HO.svg"
plot_predictions_vs_truth(prefix,gt_utils, save_dir,save_name=save_name,scaler=None, show_n_indv=5)