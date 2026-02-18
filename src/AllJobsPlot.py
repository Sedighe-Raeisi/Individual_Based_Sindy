import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors

# Global list of distinct colors for jobs
# We can use a palette to ensure different jobs have distinct colors
JOB_COLORS = list(mcolors.TABLEAU_COLORS.values())
JOB_LINE_STYLE = ['solid', 'dashed', 'dashdot', 'dotted']


def Custom_plot_combined(axi, job_path, job_index, generate_pdf, ground_truth=True, HB_Est=True,
                         pdf_state=False, gt_utils=None, realparame2gtarray=None, scaler=None,
                         to_plot_idx=None, plot_dict=dict(), xlim=None, y_lim_max_ref=0):
    """
    Plots the HB estimate from a single job onto an existing matplotlib Axes object (axi).

    Args:
        axi (plt.Axes): The axes object to plot onto.
        job_path (str): The full path to the current job directory.
        job_index (int): Index of the current job (used for color/label).
        to_plot_idx (list): The [eq_i, coef_i] index for the coefficient being plotted on this axi.
        y_lim_max_ref (float): The current maximum density height across all plots.
        # Other arguments are similar to the original Custom_plot but simplified.
    """

    # Extract plot parameters from plot_dict
    xlabel_fontsize = plot_dict.get('xlabel_fontsize', None)
    gt_color = plot_dict.get('gt_color', 'red')
    pdf_color = plot_dict.get("pdf_color", "black")
    pdf_fill = plot_dict.get("pdf_fill", False)

    current_color = JOB_COLORS[job_index % len(JOB_COLORS)]
    current_linestyle = JOB_LINE_STYLE[job_index % len(JOB_LINE_STYLE)]
    current_label = f"HB Est. (Job {job_index + 1})"

    y_lim = y_lim_max_ref
    true_params_file_str = f"chk_GT_Data.pkl"

    ############################## Load GT Data (Only for the first job to set GT/PDF) ###########################
    # The GT and PDF data are the same across all jobs, so we only need to load them once.
    if job_index == 0:
        true_params_filename = os.path.join(job_path, true_params_file_str)
        try:
            with open(true_params_filename, 'rb') as f:
                X_data, Y_data, true_params = pickle.load(f)
            gt_coef_array = realparame2gtarray(true_params)

            # --- Plot Ground Truth (Observed Data) ---
            if ground_truth:
                print("gt add")
                sns.kdeplot(gt_coef_array[to_plot_idx[0], to_plot_idx[1], :], ax=axi, fill=False, color=gt_color,
                            alpha=.8,
                            warn_singular=False, linewidth=1, label="Observed Data")

                data = gt_coef_array[to_plot_idx[0], to_plot_idx[1], :]
                peak_y = np.max(gaussian_kde(data)(np.linspace(min(data), max(data), 100)))
                y_lim = max(y_lim, peak_y)

            # --- Plot PDF ---
            if pdf_state:
                pdf_arr = generate_pdf(job_path)
                print("pdf add")
                sns.kdeplot(pdf_arr[to_plot_idx[0], to_plot_idx[1], :], ax=axi, fill=pdf_fill, color=pdf_color,
                            alpha=.4,
                            warn_singular=False, linewidth=1, label="Data Generating PDF")

                data = pdf_arr[to_plot_idx[0], to_plot_idx[1], :]
                peak_y = np.max(gaussian_kde(data)(np.linspace(min(data), max(data), 100)))
                y_lim = max(y_lim, peak_y)

        except Exception as e:
            print(f"Error loading GT/PDF data for job {job_index + 1}: {e}")
            return y_lim_max_ref

    ########################## Load HB EST data for current job #############################
    try:
        if HB_Est:
            if scaler is None:
                npz_sample_path = os.path.join(job_path, "mcmc_samples.npz")
                loaded_samples = np.load(npz_sample_path, allow_pickle=True)
                est_coef = loaded_samples['coef']
                est_coef_array = np.array(est_coef)
            else:
                with open(os.path.join(job_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
                    mcmc_coef_results = pickle.load(f)
                    est_coef_array = np.array(mcmc_coef_results)

            # HB Estimate is usually the mean of the coefficient across MCMC samples
            est_coef_mean_arr = np.mean(est_coef_array, axis=0)

            # --- Plot HB Estimate for current job ---
            sns.kdeplot(est_coef_mean_arr[:, to_plot_idx[1], to_plot_idx[0]], ax=axi, fill=False, color=current_color,
                        alpha=.6,
                        warn_singular=False, linewidth=1, ls=current_linestyle, label=current_label)

            data = est_coef_mean_arr[:, to_plot_idx[1], to_plot_idx[0]]
            peak_y = np.max(gaussian_kde(data)(np.linspace(min(data), max(data), 100)))
            y_lim = max(y_lim, peak_y)

    except Exception as e:
        print(f"Error loading or plotting HB data for job {job_index + 1}: {e}")

    # No axis customization here, just return the max density found
    return max(y_lim, y_lim_max_ref)


def all_plot_combined(all_plot_dict):
    rootdir = all_plot_dict.get("Bh_path", None)
    generate_pdf = all_plot_dict.get("generate_pdf", None)
    gt_utils = all_plot_dict.get("gt_utils", None)
    realparame2gtarray = all_plot_dict.get("realparame2gtarray", None)
    # flat_save_path = all_plot_dict.get("flat_save_path", None) # Not used in combined plot logic
    scaler = all_plot_dict.get("scaler", None)
    to_plot = all_plot_dict.get("to_plot", None)
    plot_dict = all_plot_dict.get("plot_dict", None)
    n_rows = all_plot_dict.get("n_rows", None)
    n_cols = all_plot_dict.get("n_cols", None)
    xlim = all_plot_dict.get("xlim", None)
    plot_name = plot_dict.get("plot_name", 'LV_allJobs')  # Get name from plot_dict or use default
    xlabel_list = plot_dict.get("xlabel_list", None)
    xlabel_fontsize = plot_dict.get('xlabel_fontsize', None)
    legend_True = plot_dict.get('legend', False)
    fighigth = all_plot_dict.get("fighigth", 3)
    figwidth = all_plot_dict.get("figwidth", 12)
    pdf_state = all_plot_dict.get("pdf_state", True)  # Assuming you want PDF now

    if not rootdir:
        print("Error: Bh_path (rootdir) not provided.")
        return

    # 1. Prepare Figure and Axes
    fig = plt.figure(layout="constrained", figsize=(figwidth, fighigth * n_rows))

    # Store the Axes objects in a list
    axes = []
    plot_y_limits = [0] * len(to_plot)  # Track max density for each subplot

    for i in range(len(to_plot)):
        # Create subplots for each coefficient to plot
        axi = plt.subplot2grid((n_rows, n_cols), (i // n_cols, i % n_cols), colspan=1, fig=fig)
        axes.append(axi)

        # --- Axis Customization (from original Custom_plot) ---
        axi.set_ylabel(" ", fontsize=xlabel_fontsize)
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
        axi.set_yticks(np.array([0]))
        axi.set_yticklabels([""], fontsize=12, rotation=90)
        axi.yaxis.set_tick_params(length=0)
        axi.xaxis.set_tick_params(labelsize=xlabel_fontsize - 4 if xlabel_fontsize else 8)
        axi.spines['left'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)

        # Set X-label
        if xlabel_list and i < len(xlabel_list):
            axi.set_xlabel(xlabel_list[i], fontsize=xlabel_fontsize)
        # else:
    # Requires loading GT data once to get coef_names and eqs for the label
    # This is complex, better to rely on xlabel_list or define labels later.
    # For simplicity, we skip automatic labeling here.

    # 2. Iterate over all job directories and plot on the existing axes
    all_jobs_path = os.listdir(rootdir)
    job_paths = [os.path.join(rootdir, job_file) for job_file in all_jobs_path if
                 os.path.isdir(os.path.join(rootdir, job_file))]

    # Filter to ensure we only process job directories (e.g., LV_chk_..._job1)
    job_paths.sort()  # Ensure job1, job2, ... order

    print(f"Found {len(job_paths)} job directories to combine plots.")

    # Loop over the found job directories
    for job_index, job_path in enumerate(job_paths):

        # Loop over the coefficients we want to plot
        for plot_i, to_plot_idx in enumerate(to_plot):
            # Call the combined plotter for the current subplot (axes[plot_i])
            # The function handles plotting the GT/PDF data only on the first job (index 0)
            print("+++++++++++++++++++++++++++++++++++++++")
            print("job_path = ",job_path)

            print("ax_i = ",plot_i)
            max_y = Custom_plot_combined(
                axes[plot_i],
                job_path,
                job_index,
                generate_pdf,
                ground_truth=True,
                HB_Est=True,
                pdf_state=pdf_state,
                gt_utils=gt_utils,
                realparame2gtarray=realparame2gtarray,
                scaler=scaler,
                to_plot_idx=to_plot_idx,
                plot_dict=plot_dict,
                xlim=xlim,
                y_lim_max_ref=plot_y_limits[plot_i]
            )
            # Update the maximum density height for this subplot
            plot_y_limits[plot_i] = max_y

    # 3. Finalize Plot (Set Y-limits and Legends)
    for plot_i, axi in enumerate(axes):
        # Set the Y-limit based on the maximum density found across ALL jobs/data types
        axi.set_ylim(0, plot_y_limits[plot_i] * 1.2)

        # Add legend to the last subplot if required
        if legend_True and plot_i == len(to_plot) - 1:
            # We place the legend outside the last plot
            axi.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                       fontsize=xlabel_fontsize - 4 if xlabel_fontsize else 8)

    # 4. Save and Show
    plt.subplots_adjust(
        left=0.05,  # Left margin
        right=0.9,  # Right margin (pulls subplots left)
        bottom=0.2,  # Bottom margin (pushes subplots up)
        wspace=0.6  # Width space between subplots
    )

    # Save the figure
    # Note: Using the plot_name from the dictionary, assuming it's correctly set.
    plt.savefig(os.path.join(os.getcwd(), plot_name) + ".svg")
    plt.show()
    print("Combined plot run finished")