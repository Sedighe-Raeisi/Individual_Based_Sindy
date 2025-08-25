from src.compare_0 import compare_model
save_dir_prefix = "FHN_chk_"
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
compare_model(save_dir_prefix)