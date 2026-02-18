
from src.model import MultiTargetMultiEquation_HSModel
from src.overfit_report import path2idata, _generate_overfitting_report
from src.compare import compare_model
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

print("**************** Evaluation of model based on LOO metric for overfit *******************")
hb_save_dir_prefix = "RLC_chk_"
root_path = os.getcwd()
idata = path2idata(root_path,hb_save_dir_prefix,scaler=None)
_generate_overfitting_report(idata,MultiTargetMultiEquation_HSModel)

print("************ Model Comparison Evaluation ************")
compare_model(hb_save_dir_prefix)

