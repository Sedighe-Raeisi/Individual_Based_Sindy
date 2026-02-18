import os
import numpy as np
import re # Required for load_and_average_HB_from_jobs
import pickle # Required for load_and_average_HB_from_jobs

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

# Assuming Custom_plot is now updated to accept 'combined_HB_samples' and the loading function is available.
# Since you provided Custom_plot code previously, I'll assume the updated version is imported here.
from src.custom_plot import Custom_plot 
from src.Dynamical_systems_utils.Lotka_Volterra.Lotka_Volterra import gt_utils,realparame2gtarray,generate_pdf

# --- REQUIRED UTILITY FUNCTION: load_and_average_HB_from_jobs (Copied here for completeness) ---

def load_and_average_HB_from_jobs(base_dir, save_dir_prefix, max_repeats):
    """Loads 'mcmc_samples.npz' from all job iteration directories and combines them."""
    all_coef_samples = []
    parent_archive_dir = os.path.join(base_dir, f"chk_{save_dir_prefix.split('_')[0]}_dir")
    
    print(f"Searching for job directories in: {parent_archive_dir}")

    for i in range(1, max_repeats + 1):
        job_suffix = f"_job{i}"
        job_dir_name = None
        # Find the specific directory inside the parent_archive_dir
        for dir_name in os.listdir(parent_archive_dir):
            if dir_name.startswith(save_dir_prefix) and dir_name.endswith(job_suffix):
                # Ensure the pattern between prefix and suffix is date/time like (e.g., LV_chk_YYYYMMDD_HHMM_jobN)
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
                    if 'coef' in loaded_samples:
                        all_coef_samples.append(np.array(loaded_samples['coef']))
                except Exception as e:
                    print(f"Error loading {npz_sample_path}: {e}")
            else:
                print(f"Warning: File not found at {npz_sample_path}")
        else:
            print(f"Warning: Directory with prefix '{save_dir_prefix}' and suffix '{job_suffix}' not found in {parent_archive_dir}")

    if not all_coef_samples:
        raise ValueError("No valid coefficient samples were loaded from any job iteration.")

    combined_samples = np.concatenate(all_coef_samples, axis=0)
    print(f"Successfully combined samples. Total shape: {combined_samples.shape}")
    return combined_samples

# -------------------------------------------------------------------------------------


# --- EXECUTION PARAMETERS ---
save_dir_prefix = "LV_chk"
MAX_REPEATS = 4 # Total number of jobs (job1 through job4)
HOME_DIR = "/home/staff/s/sraeisi" # Base directory for the archive folder 'chk_LV_dir'

to_plot = [[0,1],[0,3],[1,2],[1,3]]
xlabel_list = ["$\\dot{x}$ : $x$","$\\dot{x}$ : $x\\cdot y$","$\\dot{y}$ : $y$","$\\dot{y}$ : $x\\cdot y$"]

plot_dict = {"legend":True, "est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"black",
             "xlabel_fontsize": 24, "title_fontsize": None,
             "max_y_limit": 40.3,"plot_name":"plot_LV_Combined_HB","pdf_fill":False,"xlabel_list":xlabel_list}


# --- NEW: LOAD COMBINED HB SAMPLES ---
try:
    combined_hb_data = load_and_average_HB_from_jobs(
        base_dir=HOME_DIR,
        save_dir_prefix=save_dir_prefix,
        max_repeats=MAX_REPEATS
    )
except Exception as e:
    print(f"Fatal Error during data loading: {e}")
    exit(1)


# --- EXECUTE PLOTTING WITH COMBINED DATA ---
Custom_plot(
    generate_pdf, 
    pdf_state=True, 
    ground_truth = True, 
    HB_Est = True, 
    FlatB_Est = False, # Set to False, assuming only the combined HB is required
    TABLE = False,
    gt_utils=gt_utils, 
    realparame2gtarray=realparame2gtarray, 
    save_dir_prefix=save_dir_prefix,
    fighigth=3, 
    figwidth=12, 
    n_rows=1, 
    n_cols=4,
    scaler="scale", 
    to_plot=to_plot, 
    plot_dict=plot_dict,
    # This is the crucial addition: passing the combined data to Custom_plot
    combined_HB_samples=combined_hb_data 
)