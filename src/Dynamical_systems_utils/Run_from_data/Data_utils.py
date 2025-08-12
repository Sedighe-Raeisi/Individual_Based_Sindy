import numpy as np
import time
from typing import Callable
import pandas as pd
import math
import pickle

# from src.Dynamical_systems_utils.Cognetive_RL.Daniel_Code.datasize_analysis import forget_rate
from src.utils import gen_param

############################### Mix data ######################################
def mix_data(system_param_dict):
    data_path = system_param_dict["data_path"]
    with open(data_path, 'rb') as f:
        XMixedData_np, YMixedData_np = pickle.load(f)

    print("===============Preparing mix data is complete===============")
    real_params = None



    return XMixedData_np, YMixedData_np, real_params


def gt_utils(real_params):
    eqs = ["QA_t = QA_[t-1]+Action*Alpha*(R_QA[t-1])+(1-Action)*ForgetRate*(Q0-QA[t-1])"]
    coef_names = ['QA_1','ActionA*(R-QA_1)','(1-ActionA)*(QA0-QA_1)','QA_1**2','QB_1','QA_1*QB_1'] # Updated coef_names for RLC

    gt_coef = [{'QA_1': ['-', '-'],
                'ActionA*(R-QA_1)': ['-', '-'],
                '(1-ActionA)*(QA0-QA_1)': ['-', '-'],
                'QA_1**2':['-','-'],
                'QB_1':['-','-'],
                'QA_1*QB_1':['-','-']}]
    return {"eqs":eqs, "coef_names":coef_names,"gt_coef":gt_coef}

def realparame2gtarray(real_params):
    pass

def generate_pdf(save_path, pdf_smaple_N=10000, epsilon = 0.001 ):
    pass
