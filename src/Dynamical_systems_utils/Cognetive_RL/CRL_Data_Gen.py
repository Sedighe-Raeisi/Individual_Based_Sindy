import numpy as np
import time
from typing import Callable
from src.Dynamical_systems_utils.Cognetive_RL.Daniel_Code.resources import bandits
import pandas as pd
import math
import pickle

# from src.Dynamical_systems_utils.Cognetive_RL.Daniel_Code.datasize_analysis import forget_rate
from src.utils import gen_param

def data_generator(

  n_trials_per_session = 64,
  n_sessions = 128,

  # ground truth parameters
  alpha = 0.27,
  beta = 3,
  forget_rate = 0.,
  perseveration_bias = 0.,
  correlated_update = False,
  regret = False,
  confirmation_bias = False,
  reward_update_rule: Callable = None,

  # environment parameters
  n_actions = 2,
  sigma = 0.2,
  correlated_reward = False,
  non_binary_reward = False,
  ):
    time_str = int(time.time())
    # setup
    environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_reward=non_binary_reward, correlated_reward=correlated_reward)
    agent = bandits.AgentQ(alpha, beta, n_actions, forget_rate, perseveration_bias, correlated_update, regret, confirmation_bias)
    if reward_update_rule is not None:
        agent.set_reward_update(reward_update_rule)
    print('Setup of the environment and agent complete.')

    print('Creating the test dataset...', end='\r')
    dataset, experiment_list = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions)

    print('Setup of datasets complete.')

    choices = []
    rewards = []
    qs = []
    reward_probs = []
    probs = []

    for session_id in range(len(experiment_list)):
      choices.append(experiment_list[session_id].choices)
      rewards.append(experiment_list[session_id].rewards)

      list_probs = []
      list_qs = []

      # get q-values from groundtruth
      qs_test, probs_test = bandits.get_update_dynamics(experiment_list[session_id], agent)
      list_probs.append(np.expand_dims(probs_test, 0))
      list_qs.append(np.expand_dims(qs_test, 0))

      # concatenate all choice probs and q-values
      probs_id = np.concatenate(list_probs, axis=0)
      qs_id = np.concatenate(list_qs, axis=0)

      probs.append(probs)
      qs.append(qs_id)

      reward_probs.append(np.stack([experiment_list[session_id].timeseries[:, i] for i in range(n_actions)], axis=0))

    return rewards, qs, choices

############################## Data Preprocessor ###################################
def preprocess_data(qs_list, rewards_list, choices_list,alpha,forget_rate):
  DF=pd.DataFrame()

  for i in range(len(qs_list)):

    qs =  qs_list[i]
    rewards = rewards_list[i]
    choices = choices_list[i]

    qs=np.squeeze(qs)
    rewards=np.squeeze(rewards)
    choices=np.squeeze(choices)


    df=pd.DataFrame()



    df['QA_1']=qs[:,0]
    df['QB_1']=qs[:,1]
    df['R']=rewards
    df['Choice']=choices
    df['ActionA']=1-df['Choice']
    df['ActionB']=df['Choice']




    df['R-QA_1']=df['R']-df['QA_1']
    df['R-QB_1']=df['R']-df['QB_1']

    df['QA0']=df['QA_1'].iloc[0]
    df['QB0']=df['QB_1'].iloc[0]

    df['QA0-QA_1']=df['QA0']-df['QA_1']
    df['QB0-QB_1']=df['QB0']-df['QB_1']

    df['ActionA*(R-QA_1)']=df['ActionA']*df['R-QA_1']
    df['ActionB*(R-QB_1)']=df['ActionB']*df['R-QB_1']

    df['(1-ActionA)*(QA0-QA_1)']=(1-df['ActionA'])*df['QA0-QA_1']
    df['(1-ActionB)*(QB0-QB_1)']=(1-df['ActionB'])*df['QB0-QB_1']

    df['QA_1*QB_1']=df['QA_1']*df['QB_1']

    df['QA']=df['QA_1'].shift(-1)
    df['QB']=df['QB_1'].shift(-1)
    df['alpha'] = alpha
    df['forget_rate'] = forget_rate
    df['QA_1**2']=df['QA_1'].apply(lambda x: x**2)
    df['QB_1**2'] = df['QB_1'].apply(lambda x: x ** 2)
    df['QA_1*QB_1'] = df.apply(lambda x: x['QA_1']*x['QB_1'] , axis= 1)

    df.dropna(inplace=True)
    output_df=pd.concat([DF, df[['QA','QA_1','QB_1','ActionA*(R-QA_1)','(1-ActionA)*(QA0-QA_1)','alpha','forget_rate','QA_1**2','QB_1**2','QA_1*QB_1']]])
  # time.sleep(1)
  print("===============preprocessing of data is complete==================")
  return output_df

############################### Mix data ######################################
def mix_data(system_param_dict,n_trials_per_session = None, n_sessions = None,
             XName_list=['QA_1','ActionA*(R-QA_1)','(1-ActionA)*(QA0-QA_1)','QA_1**2','QB_1','QA_1*QB_1']):

    N_param_set = system_param_dict['N_param_set']
    Alpha_V = system_param_dict['Alpha_info']["Alpha_V"] if "Alpha_V" in list(system_param_dict["Alpha_info"].keys()) else None
    Alpha_mean = system_param_dict['Alpha_info']["Alpha_mean"] if "Alpha_mean" in list(system_param_dict["Alpha_info"].keys()) else None
    Alpha_std = system_param_dict['Alpha_info']["Alpha_std"] if "Alpha_std" in list(system_param_dict["Alpha_info"].keys()) else None


    ForgetRate_V = system_param_dict['ForgetRate_info']["ForgetRate_V"] if "ForgetRate_V" in list(
        system_param_dict["ForgetRate_info"].keys()) else None
    ForgetRate_mean = system_param_dict['ForgetRate_info']["ForgetRate_mean"] if "ForgetRate_mean" in list(
        system_param_dict["ForgetRate_info"].keys()) else None
    ForgetRate_std = system_param_dict['ForgetRate_info']["ForgetRate_std"] if "ForgetRate_std" in list(
        system_param_dict["ForgetRate_info"].keys()) else None

    n_trials_per_session = system_param_dict['Session_info']["n_trials_per_session"] if "n_trials_per_session" in list(
        system_param_dict["Session_info"].keys()) else 100
    n_sessions = system_param_dict['Session_info']["n_sessions"] if "n_sessions" in list(
        system_param_dict["Session_info"].keys()) else 2




    XMixedData_list=[]
    YMixedData_list=[]
    data_i=0
    alpha_list = []
    fr_list = []
    Alpha_gen = gen_param(N_param_set, Alpha_V, Alpha_mean, Alpha_std)
    FR_gen = gen_param(N_param_set, ForgetRate_V, ForgetRate_mean, ForgetRate_std)
    for param_set in range(N_param_set):
        alpha = math.fabs(Alpha_gen.gen())
        forget_rate = math.fabs(FR_gen.gen())
        alpha_list.append(alpha)
        fr_list.append(forget_rate)
        data_i+=1
        rewards_list, qs_list, choices_list = data_generator(
            alpha=alpha,
            forget_rate=forget_rate,
            # perserveration_bias = 0.,
            # regret = False,
            n_trials_per_session=n_trials_per_session,  # 64,
            n_sessions=n_sessions)

        oneIndv_df = preprocess_data(qs_list, rewards_list, choices_list,alpha,forget_rate)


        XMixedData_list.append(oneIndv_df[XName_list].to_numpy())
        YMixedData_list.append(oneIndv_df[['QA']].to_numpy())


    XMixedData_np = np.swapaxes(np.array(XMixedData_list),1,2)
    YMixedData_np = np.swapaxes(np.array(YMixedData_list),1,2)
    print("===============Preparing mix data is complete===============")
    real_params = {'alpha_mean':np.mean(np.array(alpha_list)),
                   'alpha_std': np.std(np.array(alpha_list)),
                   'forgetrate_mean':np.mean(np.array(fr_list)),
                   'forgetrate_std': np.std(np.array(fr_list)),
                   'alpha_array': np.array(alpha_list),
                   'forgetrate_array': np.array(fr_list)}



    return XMixedData_np, YMixedData_np, real_params


def gt_utils(real_params):
    eqs = ["QA_t = QA_[t-1]+Action*Alpha*(R_QA[t-1])+(1-Action)*ForgetRate*(Q0-QA[t-1])"]
    coef_names = ['QA_1','ActionA*(R-QA_1)','(1-ActionA)*(QA0-QA_1)','QA_1**2','QB_1','QA_1*QB_1'] # Updated coef_names for RLC

    gt_coef = [{'QA_1': [1, 0],
                'ActionA*(R-QA_1)': [real_params['alpha_mean'], real_params['alpha_std']],
                '(1-ActionA)*(QA0-QA_1)': [real_params['forgetrate_mean'], real_params['forgetrate_std']],
                'QA_1**2':[0,0],
                'QB_1':[0,0],
                'QA_1*QB_1':[0,0]}]
    return {"eqs":eqs, "coef_names":coef_names,"gt_coef":gt_coef}

def realparame2gtarray(real_params:dict):

    QA_arr = np.ones_like(real_params['alpha_array'])
    alpha_arr = real_params['alpha_array']
    fr_arr = real_params['forgetrate_array']
    QA2_arr = np.zeros_like(real_params['forgetrate_array'])
    QB_arr = np.zeros_like(real_params['forgetrate_array'])
    QAQB_arr = np.zeros_like(real_params['forgetrate_array'])



    eq1_coef = [QA_arr,alpha_arr,fr_arr,QA2_arr,QB_arr,QAQB_arr]


    eqs = np.array([eq1_coef])
    return eqs

def generate_pdf(save_path, pdf_smaple_N=10000, epsilon = 0.001 ):
    pass
