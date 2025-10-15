from src.compare import compare_model
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

save_dir_prefix = "DFHO_chk_"
compare_model(save_dir_prefix)