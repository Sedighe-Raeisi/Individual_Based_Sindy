import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pickle
import os






# --- 2. Prediction and Plotting Function ---
def plot_predictions_vs_truth(prefix,gt_utils, save_dir,save_name,scaler=None, show_n_indv=5):
    """
    Generates predictions from MCMC samples and plots the true Y vs. predicted Y
    for a subset of individuals.

    Args:
        gt_utils_func (callable): Function to get model details (like equation names).
        save_dir (str): Directory to save the plots and read data.
        show_n_indv (int): Number of individuals to plot.
    """
    # 1. Get posterior samples for coefficients

    file_name = [f for f in os.listdir() if f.startswith(prefix)][0]
    save_path = os.getcwd()
    save_path = os.path.join(save_path, file_name)
    true_params_filename = os.path.join(save_path, "chk_GT_Data.pkl")
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    system_param_dict_path = os.path.join(save_path, "system_param_dict.pkl")
    with open(system_param_dict_path, "rb") as f:
        system_param_dict = pickle.load(f)

    if scaler is None:
        npz_sample_path = os.path.join(save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        est_coef = loaded_samples['coef']
        coef_samples = np.array(est_coef)
    else:
        with open(os.path.join(save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            print(f" start loading samples from {os.path.join(save_path, f'revert_mcmc_samples.pkl')}")
            mcmc_coef_results = pickle.load(f)
            coef_samples = np.array(mcmc_coef_results) # Shape (N_samples, N_IDs, N_features, N_targe




    # gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eq_list = gt_dict['eqs']
    # coef_names = gt_dict['coef_names']


    print("\n--- Starting Prediction Plotting ---")
    N_samples, N_IDs, N_features, N_targets = coef_samples.shape
    N_obs = X_data.shape[2]  # N_obs is the number of observations for each individual

    print(f"Generating predictions for {N_IDs} individuals, {N_targets} targets, and {N_samples} MCMC samples.")

    # Convert numpy arrays to jax arrays for efficient calculation
    X_data_jnp = jnp.array(X_data)

    # 2. Generate Y_est predictions for each sample
    # Use vmap to efficiently calculate y_est across all N_samples
    # y_est = jnp.einsum('ifo,ift->ito', xx, coef)
    y_est_samples = jax.vmap(lambda coef: jnp.einsum('ifo,ift->ito', X_data_jnp, coef))(coef_samples)
    # y_est_samples shape: (N_samples, N_IDs, N_targets, N_obs)

    # 3. Calculate Prediction Statistics (Mean and CI)
    mean_y_est = jnp.mean(y_est_samples, axis=0)  # (N_IDs, N_targets, N_obs)
    lower_ci = jnp.percentile(y_est_samples, 2.5, axis=0)
    upper_ci = jnp.percentile(y_est_samples, 97.5, axis=0)

    # 4. Determine which individuals to plot
    plot_IDs = np.arange(N_IDs)
    if N_IDs > show_n_indv:
        # Select first 'show_n_indv' individuals
        plot_IDs = plot_IDs[:show_n_indv]

    # Get equation strings for titles (assuming gt_utils_func can work with an empty dict)


    # 5. Plotting
    print("Starting plot generation...")
    for target_idx in range(N_targets):
        eq_str = eq_list[target_idx]

        fig, axes = plt.subplots(len(plot_IDs), 1, figsize=(10, 3 * len(plot_IDs)), sharex=True)
        if len(plot_IDs) == 1:
            axes = [axes]  # Ensure axes is iterable even for a single subplot

        for i, indv_id in enumerate(plot_IDs):
            ax = axes[i]

            # True Y data for this individual and target
            true_y = Y_data[indv_id, target_idx, :]

            # X-axis is the observation index 't'
            x_axis = np.arange(N_obs)

            # Plot True Y
            ax.plot(x_axis, true_y, 'k-', label='True Y', linewidth=2)

            # Plot Mean Predicted Y
            mean_pred = mean_y_est[indv_id, target_idx, :]
            ax.plot(x_axis, mean_pred, 'r--', label='Mean Prediction', linewidth=1.5)

            # Plot Credible Interval (Uncertainty)
            lower = lower_ci[indv_id, target_idx, :]
            upper = upper_ci[indv_id, target_idx, :]
            ax.fill_between(x_axis, lower, upper, color='r', alpha=0.15, label='95% CI')

            ax.set_title(f'Individual {indv_id + 1} | {eq_str}', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Finalize plot for this target
        axes[-1].set_xlabel("Observation Index (t)")

        # Add a common Y-label (approximate center)
        # if len(plot_IDs) > 0:
        #     axes[len(plot_IDs) // 2].set_ylabel("Y / $\hat{Y}$ Value")

        # Add legend only once on the last subplot
        axes[0].legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        # Save the plot

        plot_filename = os.path.join(save_dir, f"{target_idx + 1}_"+save_name)
        plt.savefig(plot_filename)
        print(f"✅ Plot saved: {plot_filename}")
        plt.close(fig)
        print("plot generation end")

    # --- 3. Main Execution Block ---


