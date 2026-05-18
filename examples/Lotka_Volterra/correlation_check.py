import numpy as np
import pickle
import os
from src.Dynamical_systems_utils.Lotka_Voltera import gt_utils
# from examples.LV_SINDyPI.LVM_SINDyPI_Not0Target import root_path

save_dir_prefix = "LV_chk_"
root_path = os.getcwd()
entries = os.listdir(root_path)
found_folders = []
for entry in entries:
    full_path = os.path.join(root_path, entry)
    # Check if it's a directory and starts with "chk_" (updated from "2025" for flexibility) #:
    if entry.startswith(save_dir_prefix):  #:
        program_state = "continue"
        found_folders.append(full_path)
        chk_path = os.path.join(root_path, found_folders[-1])
true_params_filename = os.path.join(chk_path, f"chk_GT_Data.pkl")
with open(true_params_filename, 'rb') as f:
    X_data, Y_data, true_params = pickle.load(f)
# Loop through your individuals (using 4 as in your example)
for i in range(1):
    # Slice shape: (10, 1000) -> 10 features, 1000 observations
    indv_data = X_data[i, :, :]

    # We remove rowvar=False because features are already the rows
    corr_matrix = np.corrcoef(indv_data)

    print(f"Individual {i} Correlation Matrix Shape: {corr_matrix.shape}")
    print(corr_matrix)
    print("-" * 30)
gt_dict = gt_utils(true_params)
eqs = gt_dict['eqs']
coef_names = gt_dict['coef_names']
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Let's use your Individual 0 data
# I'll fill the NaNs with 0 just for the sake of the visual
corr_matrix = np.corrcoef(X_data[0, :, :])
N_feature = corr_matrix.shape[1]
corr_matrix = np.nan_to_num(corr_matrix)

plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(corr_matrix,
            annot=True,          # Write the numbers in the cells
            cmap='coolwarm',     # Red for positive, Blue for negative
            fmt=".2f",           # 2 decimal places
            vmin=-1, vmax=1,     # Ensure the color scale is always -1 to 1
            # xticklabels=[f'Feat {i}' for i in range(N_feature)],
            # yticklabels=[f'Feat {i}' for i in range(N_feature)])
            xticklabels=[f'{coef_names[i]}' for i in range(N_feature)],
            yticklabels=[f'{coef_names[i]}' for i in range(N_feature)])

plt.title("Correlation Matrix for Individual 0")
plt.savefig("correlation_matrix.svg")
plt.show()
