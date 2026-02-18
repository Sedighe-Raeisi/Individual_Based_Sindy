import numpy as np
import arviz as az
import pandas as pd
import pickle
import os
from src.mcmc_utils import BH_scaler

def path2idata(root_path,hb_save_dir_prefix,scaler=None):
    # --- Process the Hierarchical Bayesian (HB) Model ---

    folder_path_hb = os.path.join(
        root_path,
        [file for file in os.listdir(root_path) if file.startswith(hb_save_dir_prefix)][0]
    )
    with open(os.path.join(folder_path_hb, "chk_GT_Data.pkl"), 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)
        if scaler is not None:
            scaler = BH_scaler(X_data)
            X_data = scaler.scale(X_data)
    with open(os.path.join(folder_path_hb, "mcmc_module.pkl"), "rb") as f:
        hb_loaded_mcmc = pickle.load(f)
    dims_hb = {"obs": ["id", "target", "time"]}
    idata_kwargs_hb = {"dims": dims_hb, "constant_data": {"xx": X_data}}
    idata_hb = az.from_numpyro(hb_loaded_mcmc, **idata_kwargs_hb)
    return idata_hb
#######################################################

def _generate_overfitting_report(idata, model_name):
    """
    Computes and reports detailed evidence of overfitting based on
    the Pareto k diagnostic (k_hat) from LOO-CV.
    """
    print(f"\n--- Overfitting Evidence Report for: {model_name} ---")


    # Compute LOO and retrieve k values
    loo_result = az.loo(idata, pointwise=True)
    k_values = loo_result.pareto_k
    k_array = k_values.values.flatten()  # Flatten k values for summary stats

    # 1. Summary Statistics for Pareto k
    k_summary = pd.Series(k_array).describe(percentiles=[.7, .9, .99])
    print("\\nStatistical Summary of Pareto k ($\\hat{k}$):")
    print(k_summary.to_markdown(floatfmt=".3f"))

    # 2. Count of Problematic Observations (The direct evidence of instability/overfitting)
    n_total = k_values.size

    # Thresholds for problematic observations (following ArviZ/PSIS guidelines)
    k_good_threshold = 0.5
    k_problematic_threshold = 0.7
    k_severe_threshold = 1.0

    # Counts
    num_k_good = (k_array < k_good_threshold).sum()
    num_k_ok = ((k_array >= k_good_threshold) & (k_array < k_problematic_threshold)).sum()
    num_k_problematic = ((k_array >= k_problematic_threshold) & (k_array < k_severe_threshold)).sum()
    num_k_severe = (k_array >= k_severe_threshold).sum()

    print("\\nPareto $\\hat{k}$ Distribution Analysis (Evidence of Overfitting/Misspecification):")
    report_data = {
        'Category': ['Total Observations', 'Good (k < 0.5)', 'High Variance (0.5 <= k < 0.7)',
                     'Problematic (0.7 <= k < 1.0)', 'Severe/Unreliable (k >= 1.0)'],
        'Count': [n_total, num_k_good, num_k_ok, num_k_problematic, num_k_severe],
        '% of Total': [100.0,
                       num_k_good / n_total * 100,
                       num_k_ok / n_total * 100,
                       num_k_problematic / n_total * 100,
                       num_k_severe / n_total * 100]
    }
    k_report_df = pd.DataFrame(report_data)
    print(k_report_df.to_markdown(index=False, floatfmt=(".0f", ".0f", ".1f")))

    # 3. Interpretation
    if num_k_severe > 0:
        print(
            f"\n🚨 **Severe Overfitting/Misspecification Warning:** {num_k_severe} ({report_data['% of Total'][4]:.1f}%) observations have k-hat $\\ge 1.0$. This means the LOO estimate is unreliable, and these points are effectively **outliers** that are disproportionately influencing the posterior. This is the **strongest evidence** of overfitting to these specific points.")
    elif num_k_problematic > 0:
        print(
            f"\n⚠️ **Moderate Overfitting/Instability Warning:** {num_k_problematic} ({report_data['% of Total'][3]:.1f}%) observations have $0.7 \\le$ k-hat $< 1.0$. These points are highly influential. While the LOO estimate is still usable, the model is showing signs of **instability** and may be overfitting these data points.")
    else:
        print(
            "\n✅ **LOO Stability:** All k-hat $< 0.7$. The model's cross-validation is stable, suggesting that it's not severely overfitting individual observations based on the PSIS diagnostic.")

    return k_report_df.set_index('Category').T

