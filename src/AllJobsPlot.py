import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from scipy.stats import gaussian_kde
import re # Import regex for pattern matching

#########################################################################
## NEW FUNCTION TO LOAD AND AVERAGE DATA ACROSS MULTIPLE JOB ITERATIONS #
#########################################################################

def load_and_average_HB_from_jobs(base_dir, save_dir_prefix, max_repeats):
    """
    Loads 'mcmc_samples.npz' from all job iteration directories, combines the
    coefficient samples, and returns the concatenated array.

    Args:
        base_dir (str): The path to the parent directory containing the 'chk_LV_dir' etc.
                        (e.g., "/home/staff/s/sraeisi/")
        save_dir_prefix (str): The prefix used for the checkpoint directory (e.g., "LV_chk").
        max_repeats (int): The total number of job iterations (e.g., 4).

    Returns:
        np.ndarray: Concatenated coefficient samples (HB_est) from all job iterations.
    """
    all_coef_samples = []
    
    # Construct the parent archive directory name (e.g., chk_LV_dir)
    parent_archive_dir = os.path.join(base_dir, f"chk_{save_dir_prefix.split('_')[0]}_dir")
    
    print(f"Searching for job directories in: {parent_archive_dir}")

    # Loop through job iterations (1 to max_repeats)
    for i in range(1, max_repeats + 1):
        job_suffix = f"_job{i}"
        
        # We need to find the specific directory inside the parent_archive_dir
        # that ends with the job_suffix (e.g., LV_chk_20251119_1600_job1)
        job_dir_name = None
        for dir_name in os.listdir(parent_archive_dir):
            if dir_name.startswith(save_dir_prefix) and dir_name.endswith(job_suffix):
                 # Use regex to ensure the pattern between prefix and suffix is date/time like
                if re.match(r'^' + re.escape(save_dir_prefix) + r'_\d{8}_\d{4}' + re.escape(job_suffix) + r'$', dir_name):
                    job_dir_name = dir_name
                    break
        
        if job_dir_name:
            job_path = os.path.join(parent_archive_dir, job_dir_name)
            npz_sample_path = os.path.join(job_path, "mcmc_samples.npz")
            
            if os.path.exists(npz_sample_path):
                print(f"Loading samples from: {npz_sample_path}")
                try:
                    loaded_samples = np.load(npz_sample_path, allow_pickle=True)
                    # Check for 'coef' key in the loaded .npz file
                    if 'coef' in loaded_samples:
                        coef_samples = loaded_samples['coef']
                        # Ensure samples are correctly shaped as a NumPy array
                        all_coef_samples.append(np.array(coef_samples))
                    else:
                        print(f"Warning: 'coef' key not found in {npz_sample_path}")
                except Exception as e:
                    print(f"Error loading {npz_sample_path}: {e}")
            else:
                print(f"Warning: File not found at {npz_sample_path}")
        else:
            print(f"Warning: Directory with prefix '{save_dir_prefix}' and suffix '{job_suffix}' not found in {parent_archive_dir}")

    if not all_coef_samples:
        raise ValueError("No valid coefficient samples were loaded from any job iteration.")

    # Concatenate all samples along the first axis (MCMC chain length)
    combined_samples = np.concatenate(all_coef_samples, axis=0)
    print(f"Successfully combined samples. Total shape: {combined_samples.shape}")
    return combined_samples

#########################################################################
## MODIFIED Custom_plot FUNCTION (to integrate the new loader) #
#########################################################################

# Your existing Custom_plot function (slightly adapted for clarity and new input)

# inputs:
    # which indices(coef, eq),
    # how many row,col,
    # (gt, HB, Flat_B)
    # with or without table?

def Custom_plot(generate_pdf, ground_truth = True, HB_Est = True, FlatB_Est = True,pdf_state = True, TABLE = False, gt_utils=None, realparame2gtarray=None, save_dir_prefix=None,
               fighigth=3, figwidth=12, n_rows=None, n_cols=None,
               scaler=None, to_plot=None, plot_dict=dict(),xlim=None,
               # New argument for combined HB samples
               combined_HB_samples=None): 
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams['font.family'] = 'STIXGeneral'


    # Add an explicit setting for the axes label font properties if needed
    plt.rcParams['axes.labelsize'] = 'large'
    if TABLE:
        all_cols = n_cols+2
    else:
        all_cols = n_cols

    root_path = os.getcwd()
    true_params_file_str = f"chk_GT_Data.pkl"
    
    # Determine the save_path: we just need one directory to find the GT data, 
    # but the HB data is passed via combined_HB_samples.
    # We'll use the first job's archive directory to find the GT data file.
    # We must adapt this to find the *actual* path to the first job directory inside the archive.
    
    # We'll assume the GT data is stored in the first job's directory, which should be consistent.
    # The current save_dir_prefix is 'LV_chk'. We need to find the directory starting with it.
    
    # Find the parent archive dir path (e.g., chk_LV_dir)
    parent_archive_dir_name = f"chk_{save_dir_prefix.split('_')[0]}_dir"
    parent_archive_path = os.path.join(root_path, parent_archive_dir_name)

    # Find the actual path to the first job's directory (e.g., /.../chk_LV_dir/LV_chk_20251119_1600_job1)
    save_path = None
    for dir_name in os.listdir(parent_archive_path):
        if dir_name.startswith(save_dir_prefix) and dir_name.endswith("_job1"):
            save_path = os.path.join(parent_archive_path, dir_name)
            break
            
    if not save_path:
        raise FileNotFoundError(f"Could not find a starting job directory (e.g., {save_dir_prefix}_..._job1) in {parent_archive_path}")

    legend_True = plot_dict.get('legend', False)
    xlabel_fontsize = plot_dict.get('xlabel_fontsize', None)
    title_fontsize = plot_dict.get('title_fontsize', None)
    est_color = plot_dict.get('est_color', 'blue')
    gt_color = plot_dict.get('gt_color', 'pink')
    flat_color = plot_dict.get('flat_color', 'grey')
    pdf_color = plot_dict.get("pdf_color","red")
    pdf_fill = plot_dict.get("pdf_fill",False)
    pdf_line_width = plot_dict.get("pdf_line_width",2)
    max_y_limit = plot_dict.get("max_y_limit",0.98)
    plot_name = plot_dict.get("plot_name",'custom_plot')
    xlabel_list = plot_dict.get("xlabel_list",None)
    # plt.rcParams["font.family"] = "Times New Roman"
    ############################# PDF #####################################
    if pdf_state:
        # Note: If generate_pdf needs access to the base directory of the GT data, this path must be accurate.
        pdf_arr = generate_pdf(save_path)

    ############################## Load GT Data ###########################
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    N_Indv, N_Coef, N_Obs = X_data.shape
    N_Indv, N_Eqs, N_Obs = Y_data.shape
    
    ########################## Load Combined HB EST data #############################
    # This section is simplified as we now rely on the pre-loaded combined_HB_samples
    if combined_HB_samples is None and HB_Est:
         raise ValueError("HB_Est is True but combined_HB_samples was not provided to Custom_plot.")
         
    if HB_Est:
        est_coef_array = combined_HB_samples # Use the pre-loaded, combined data
        print(f"combined_HB_samples shape = {est_coef_array.shape}")
        # est_coef_mean_arr now represents the full set of posterior samples from all jobs
        est_coef_mean_arr = est_coef_array 
        
    ########################## Load Flat EST data (Assuming FlatB still uses single job folder logic) #############################
    if FlatB_Est:
        # Replicate the logic to find the Flat_B job folder starting with 'Flat_'
        flat_save_path = None
        for dir_name in os.listdir(root_path):
            if dir_name.startswith('Flat_' + save_dir_prefix):
                flat_save_path = os.path.join(root_path, dir_name)
                break
                
        if flat_save_path:
            if scaler is None:
    
                npz_sample_path = os.path.join(flat_save_path, "mcmc_samples.npz")
                loaded_samples = np.load(npz_sample_path, allow_pickle=True)
                flat_est_coef = loaded_samples['coef']
    
                flat_est_coef_array = np.array(flat_est_coef)
            else:
                with open(os.path.join(flat_save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
                    flat_mcmc_coef_results = pickle.load(f)
                    flat_est_coef_array = np.array(flat_mcmc_coef_results)
    
            print(f"flat_coef_array.shape = {flat_est_coef_array.shape}")
            flat_est_coef_mean_arr = np.mean(flat_est_coef_array, axis=0)
        else:
            print(f"Warning: Could not find Flat Bayesian directory starting with 'Flat_{save_dir_prefix}'. Skipping FlatB_Est.")
            FlatB_Est = False

    ################### Prepare plot data and table data ##################

    mpl.rcParams['xtick.labelsize'] = 6

    missed_coef_data = []
    
    # Calculate means and stds over the combined MCMC samples for HB
    # Note: est_coef_array now holds all MCMC samples, not just means. We must compute the statistics 
    # over the final two dimensions (N_Coef, N_Eqs) after averaging/combining the MCMC chains.
    # We will use the full sample array for plotting KDE, but the table usually reports the posterior mean and std.
    
    # Calculate statistics over the MCMC chain (axis 0 is MCMC samples)
    if HB_Est:
        # Calculate mean over the MCMC samples for table
        hb_mean_per_indiv = np.mean(est_coef_array, axis=0) # Shape: (N_Indv, N_Coef, N_Eqs)
        # Calculate standard deviation over the MCMC samples for table
        hb_std_per_indiv = np.std(est_coef_array, axis=0) # Shape: (N_Indv, N_Coef, N_Eqs)


    # Identify missed coefficients and calculate their values for the table
    for eq_i in range(N_Eqs):
        for coef_i in range(N_Coef):
            if [eq_i, coef_i] not in to_plot:
                print(f"not in plot index = {[eq_i, coef_i]}")
                
                # Global mean and std for the coefficient across all individuals (HB)
                if HB_Est:
                    # We compute the global mean and std of the means across individuals for the table
                    est_mean = np.mean(hb_mean_per_indiv[:, coef_i, eq_i])
                    est_std = np.std(hb_mean_per_indiv[:, coef_i, eq_i])
                
                if FlatB_Est:
                    flatest_mean = np.mean(flat_est_coef_array[:, coef_i, eq_i])
                    flatest_std = np.std(flat_est_coef_array[:, coef_i, eq_i])

                gt_mean = np.mean(gt_coef_array[eq_i, coef_i, :])
                gt_std = np.std(gt_coef_array[eq_i, coef_i, :])
                
                row_data = [f'{eqs[eq_i].split("=")[0]} : {coef_names[coef_i]}', f'{gt_mean:.3f}']
                if HB_Est:
                    row_data.append(f'{est_mean:.3f}')
                else:
                    row_data.append('N/A')
                    
                if FlatB_Est:
                    row_data.append(f'{flatest_mean:.3f}')
                else:
                    row_data.append('N/A')
                    
                row_data.append(f'{gt_std:.3f}')
                if HB_Est:
                    row_data.append(f'{est_std:.3f}')
                else:
                    row_data.append('N/A')
                    
                if FlatB_Est:
                    row_data.append(f'{flatest_std:.3f}')
                else:
                    row_data.append('N/A')
                
                missed_coef_data.append(row_data)


    ploted_eqs = []
    plot_counter = 0
    used_axes = []
    fig = plt.figure(layout="constrained",figsize=(figwidth,fighigth))
    # Loop to create and populate the subplots for the specified indices
    plot_i = 0
    for index in to_plot:
        row_plot_idx = plot_counter // n_cols
        col = plot_counter % n_cols

        # Add a row title if it's a new row
        if index[0] not in ploted_eqs:
            # title_ax = fig.add_subplot(gs[1 * row_plot_idx, 0:n_cols])
            # title_ax.set_title(f'{eqs[index[0]]}', fontsize=12, fontweight='bold')
            # title_ax.axis('off')
            ploted_eqs.append(index[0])

        # Plot in the dedicated subplot row
        # axi = fig.add_subplot(gs[1 * row_plot_idx + 1, col])
        axi = plt.subplot2grid((n_rows, all_cols),(row_plot_idx,col),colspan=1,fig=fig)
        # used_axes.append(axi)
        y_lim = 0
        try:
            # GT Data (always plotting over the N_Indv dimension)
            if ground_truth:
                print("gt add")
                if true_params:
                    # Plot KDE over the N_Indv dimension (axis 2)
                    sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=False, color=gt_color, alpha=.8,
                                warn_singular=False, linewidth=1,label="Observed Data")

                    data = gt_coef_array[index[0], index[1], :]
                    kernel = gaussian_kde(data)
                    x_values = np.linspace(min(data), max(data), 100)
                    y_values = kernel(x_values)
                    kernel = gaussian_kde(data)
                    peak_y = np.max(y_values)
                    y_lim = max(y_lim, peak_y)
            
            # PDF Data (always plotting over the N_Indv dimension)
            if pdf_state:
                print("pdf add")
                print(pdf_arr.shape)
                # Plot KDE over the N_Indv dimension (axis 2)
                sns.kdeplot(pdf_arr[index[0], index[1],:], ax=axi, fill=pdf_fill, color=pdf_color, alpha=.4,
                            warn_singular=False, linewidth=1,label="Data Generating PDF")
                # Recalculate y_lim based on PDF for comparison if needed
                data = pdf_arr[index[0], index[1],:]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim, peak_y)
                
            # HB Estimate (NOW plotting over the combined MCMC samples dimension)
            if HB_Est:
                print("HB add")
                # Select the MCMC chain for this specific coefficient (index 0 is MCMC chain)
                # est_coef_array shape is (MCMC_samples, N_Indv, N_Coef, N_Eqs)
                # We want the posterior distribution over the individuals' means, 
                # which is the distribution of hb_mean_per_indiv[:, coef_i, eq_i] 
                # OR the distribution of the full MCMC samples for the hyper-parameters (if the data is hyper-param data)
                # Based on the original code logic, you are plotting the distribution of the individual means.
                # hb_mean_per_indiv[:, index[1], index[0]] is correct: it's the mean coefficient per individual.
                
                sns.kdeplot(hb_mean_per_indiv[:, index[1], index[0]], ax=axi, fill=False, color=est_color, alpha=.4,
                            warn_singular=False, linewidth=4,ls='--',label="Hierarchical Bayesian")
                data = hb_mean_per_indiv[:, index[1], index[0]]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim,peak_y)

        except Exception as e:
            print(f"Error during KDE plot: {e}. Falling back to old logic (if any).")
            
            # Fallback block (simplified, retaining essential KDE calls)
            if ground_truth and true_params:
                sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=False, color=gt_color, alpha=.8,
                            warn_singular=False, linewidth=1,label="Observed Data")
                data = gt_coef_array[index[0], index[1], :]
                peak_y = np.max(gaussian_kde(data)(np.linspace(min(data), max(data), 100)))
                y_lim = max(y_lim, peak_y)

            if pdf_state:
                sns.kdeplot(pdf_arr[index[0], index[1],:], ax=axi, fill=False, color=pdf_color, alpha=.4,
                            warn_singular=False, linewidth=1,label="Data Generating PDF")
                data = pdf_arr[index[0], index[1],:]
                peak_y = np.max(gaussian_kde(data)(np.linspace(min(data), max(data), 100)))
                y_lim = max(y_lim, peak_y)

            if HB_Est:
                # Still using hb_mean_per_indiv for HB plot
                sns.kdeplot(hb_mean_per_indiv[:, index[1], index[0]], ax=axi, fill=False, color=est_color, alpha=.4,
                            warn_singular=False, linewidth=4,ls='--',label="Hierarchical Bayesian")
                data = hb_mean_per_indiv[:, index[1], index[0]]
                peak_y = np.max(gaussian_kde(data)(np.linspace(min(data), max(data), 100)))
                y_lim = max(y_lim,peak_y)


        if FlatB_Est:
            print("flat add")
            print("flat shape",flat_est_coef_array.shape)
            sns.kdeplot(flat_est_coef_array[:, index[1],index[0]], ax=axi, fill=False, color=flat_color, alpha=.4,
                        warn_singular=False, linewidth=4,ls='--',label="Flat Bayesian")
            # max_y_limit = .98  # A good starting value, adjust as needed
        axi.set_ylim(0, y_lim * 1.2)

        if xlabel_list:
            axi.set_xlabel(xlabel_list[plot_i], fontsize=xlabel_fontsize)#, fontdict={'family': 'Times New Roman'})
            if xlim:
                axi.set_xlim(xlim[plot_i][0], xlim[plot_i][1])
            plot_i+=1
        else:
            axi.set_xlabel(f"{eqs[index[0]].split("=")[0]}\n{coef_names[index[1]]}", fontsize=xlabel_fontsize)#, fontdict={'family': 'Times New Roman'})
        axi.set_ylabel(" ", fontsize=xlabel_fontsize)#, fontdict={'family': 'Times New Roman'})
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
        axi.set_yticks(np.array([0]))
        axi.set_yticklabels([""], fontsize=12, rotation=90)
        axi.yaxis.set_tick_params(length=0)
        axi.xaxis.set_tick_params(labelsize=xlabel_fontsize-4)
        axi.spines['left'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)

        # axi.set_xlim(0, 2)

        if legend_True:
            if col==n_cols-1 and row_plot_idx==0:
                axi.legend(loc='center left', bbox_to_anchor=(1.0, 0.4), fontsize=xlabel_fontsize)

        plot_counter += 1

    # Hide unused subplots
    for i in range(plot_counter, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        print(row,col)
        axi = plt.subplot2grid((n_rows, all_cols), (row, col), colspan=1,fig=fig)
        axi.axis('off')

    # Add the table to the last column, spanning all 2*n_rows
    if TABLE:
        ax_table = plt.subplot2grid((n_rows, all_cols), (0, n_cols), colspan=2, rowspan=n_rows,fig=fig)
        ax_table.axis('off')

        if missed_coef_data:
            # Define table headers
            headers = ['Eq. : Coef.', 'GT\nMean', 'HB.\nMean','Flat\nMean', 'GT\nStd', 'HB.\nStd','Flat\nStd']
            # Create the table
            col_widths = [0.4, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
            table = ax_table.table(cellText=missed_coef_data,
                                   colLabels=headers,
                                   loc='center',
                                   cellLoc='center',
                                   colWidths=col_widths)
            for j in range(len(headers)):
                cell = table.get_celld()[(0, j)]
                cell.set_height(0.1)

            for i in range(len(missed_coef_data) + 1):
                for j in range(len(headers)):
                    cell = table.get_celld()[(i, j)]
                    if i == 0 or j == 0:
                        cell.set_facecolor('grey')

                    else:
                        cell.set_facecolor('white')

            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1, 1.5)
            ax_table.set_title('Other Coefficients',y=1.2, fontsize=12)

    # Use tight_layout to handle spacing between GridSpec cells
    # plt.tight_layout(w_pad=2, h_pad=2)
    plt.subplots_adjust(
        left=0.05,  # Left margin
        right=0.9,  # Right margin (pulls subplots left)
        bottom=0.2,  # Bottom margin (pushes subplots up)
        wspace=0.6  # Width space between subplots
    )
    # plt.savefig(os.path.join(os.getcwd(), plot_name)+".pdf", format="pdf")
    plt.savefig(os.path.join(os.getcwd(), plot_name) + ".svg")
    plt.show()
    print("plot run finished")