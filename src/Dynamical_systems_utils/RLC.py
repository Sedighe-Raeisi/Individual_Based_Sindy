import math
import pickle
from scipy.integrate import solve_ivp
from src.utils import gen_param
import numpy as np
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
class RLC_Circuit_Modified:
    """
    A class to simulate an RLC circuit system.
    """

    def __init__(self, L=1.0, R=1.0, C=1.0, V_in=1.0, q_0=0.0, i_0=0.0,
                 t_start=0, t_end=10, dt=0.01, noise_std=0.01):
        """Initialize the RLC circuit parameters."""
        self.L = L
        self.R = R
        self.C = C
        self.V_in = V_in
        self.q_0 = q_0
        self.i_0 = i_0
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.noise_std = noise_std

        # Time array for simulation
        self.time = np.arange(self.t_start, self.t_end, self.dt)
        self.results = {}

    def derivatives(self, t, state):
        """Define the system of differential equations for the RLC Circuit."""
        q, i = state
        dq_dt = i
        di_dt = (self.V_in - self.R * i - q / self.C) / self.L
        return [dq_dt, di_dt]

    def simulate(self):
        """Simulate the RLC Circuit and solve the differential equations."""
        state_0 = [self.q_0, self.i_0]

        # Solve the system numerically
        sol = solve_ivp(self.derivatives, [self.t_start, self.t_end], state_0, t_eval=self.time)

        # Adding Gaussian noise to charge and current
        charge_noise = np.random.normal(0, self.noise_std, size=sol.y[0].shape)
        current_noise = np.random.normal(0, self.noise_std, size=sol.y[1].shape)

        # Store the results with noise
        self.results = {
            'Time': sol.t,
            'Charge': sol.y[0] + charge_noise,
            'Current': sol.y[1] + current_noise
        }

        # Compute derivatives numerically
        dq_dt = sol.y[1]
        di_dt = np.gradient(sol.y[1], sol.t)  # More accurate computation of di/dt

        return dq_dt, di_dt
print("---------------------------- Mix Function -------------------------")
def mix_data(system_param_dict):
    """
    Generates mixed data for the RLC circuit system with varying parameters.

    Args:
        system_param_dict (dict): A dictionary containing information about the parameters
                                  (L, R, C, V_in, q0, i0) and simulation details (t_span, dt, noise_std).
                                  For each parameter (e.g., 'L_info'), it can have either a fixed value ('L_V')
                                  or information for generating samples from a normal distribution
                                  ('L_N', 'L_mean', 'L_std').

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - X_all (numpy.ndarray): Array of input features [1, q, i] for each time step and parameter set.
               - Y_all (numpy.ndarray): Array of target derivatives [dq/dt, di/dt] for each time step and parameter set.
               - real_params (dict): Dictionary containing the mean and std of the generated parameters (V/L, -R/L, -1/LC).
    """
    N_param_set = system_param_dict['N_param_set']
    L_V = system_param_dict['L_info']["L_V"] if "L_V" in list(system_param_dict["L_info"].keys()) else None
    L_N = system_param_dict['L_info']["L_N"] if "L_N" in list(system_param_dict["L_info"].keys()) else None
    L_mean = system_param_dict['L_info']["L_mean"] if "L_mean" in list(system_param_dict["L_info"].keys()) else None
    L_std = system_param_dict['L_info']["L_std"] if "L_std" in list(system_param_dict["L_info"].keys()) else None

    R_V = system_param_dict['R_info']["R_V"] if "R_V" in list(system_param_dict["R_info"].keys()) else None
    R_N = system_param_dict['R_info']["R_N"] if "R_N" in list(system_param_dict["R_info"].keys()) else None
    R_mean = system_param_dict['R_info']["R_mean"] if "R_mean" in list(system_param_dict["R_info"].keys()) else None
    R_std = system_param_dict['R_info']["R_std"] if "R_std" in list(system_param_dict["R_info"].keys()) else None
    R_2Posrtion = system_param_dict['R_info']["R_2Posrtion"] if "R_2Posrtion" in list(system_param_dict["R_info"].keys()) else None
    R_2mean = system_param_dict['R_info']["R_2mean"] if "R_2mean" in list(system_param_dict["R_info"].keys()) else None
    R_2std = system_param_dict['R_info']["R_2std"] if "R_2std" in list(system_param_dict["R_info"].keys()) else None

    C_V = system_param_dict['C_info']["C_V"] if "C_V" in list(system_param_dict["C_info"].keys()) else None
    C_N = system_param_dict['C_info']["C_N"] if "C_N" in list(system_param_dict["C_info"].keys()) else None
    C_mean = system_param_dict['C_info']["C_mean"] if "C_mean" in list(system_param_dict["C_info"].keys()) else None
    C_std = system_param_dict['C_info']["C_std"] if "C_std" in list(system_param_dict["C_info"].keys()) else None

    V_in_V = system_param_dict['V_in_info']["V_in_V"] if "V_in_V" in list(system_param_dict["V_in_info"].keys()) else None
    V_in_N = system_param_dict['V_in_info']["V_in_N"] if "V_in_N" in list(system_param_dict["V_in_info"].keys()) else None
    V_in_mean = system_param_dict['V_in_info']["V_in_mean"] if "V_in_mean" in list(
        system_param_dict["V_in_info"].keys()) else None
    V_in_std = system_param_dict['V_in_info']["V_in_std"] if "V_in_std" in list(system_param_dict["V_in_info"].keys()) else None

    q0_V = system_param_dict['q0_info']["q0_V"] if "q0_V" in list(system_param_dict["q0_info"].keys()) else None
    q0_N = system_param_dict['q0_info']["q0_N"] if "q0_N" in list(system_param_dict["q0_info"].keys()) else None
    q0_mean = system_param_dict['q0_info']["q0_mean"] if "q0_mean" in list(
        system_param_dict["q0_info"].keys()) else None
    q0_std = system_param_dict['q0_info']["q0_std"] if "q0_std" in list(system_param_dict["q0_info"].keys()) else None

    i0_V = system_param_dict['i0_info']["i0_V"] if "i0_V" in list(system_param_dict["i0_info"].keys()) else None
    i0_N = system_param_dict['i0_info']["i0_N"] if "i0_N" in list(system_param_dict["i0_info"].keys()) else None
    i0_mean = system_param_dict['i0_info']["i0_mean"] if "i0_mean" in list(
        system_param_dict["i0_info"].keys()) else None
    i0_std = system_param_dict['i0_info']["i0_std"] if "i0_std" in list(system_param_dict["i0_info"].keys()) else None


    t_span = system_param_dict['t_info']["t_span"] if "t_span" in list(system_param_dict["t_info"].keys()) else (0, 10)
    dt = system_param_dict['t_info']["dt"] if "dt" in list(system_param_dict["t_info"].keys()) else 0.01
    noise_std = system_param_dict['noise_info']["noise_std"] if "noise_std" in list(
        system_param_dict["noise_info"].keys()) else 0.01


    X_all = []
    Y_all = []

    V_L_list = []
    R_L_list = []
    rev_LC_list = []

    L_gen = gen_param(N_param_set,L_V,L_mean,L_std)
    # print(f"{N_param_set},{R_V},{R_mean},{R_std},secondpeak_posrtion= {R_2Posrtion},secondpeak_mean= {R_2mean},secondpeak_std= {R_2std}")
    R_gen = gen_param(N_param_set,R_mean,R_std,
                      secondpeak_posrtion=R_2Posrtion,secondpeak_mean=R_2mean,secondpeak_std=R_2std)
    C_gen = gen_param(N_param_set,C_V,C_mean,C_std)
    Vin_gen = gen_param(N_param_set,V_in_V,V_in_mean,V_in_std)
    q0_gen = gen_param(N_param_set,q0_V,q0_mean,q0_std)
    i0_gen = gen_param(N_param_set,i0_V,i0_mean,i0_std)
    for param_set in range(N_param_set):
        L = math.fabs(L_gen.gen())
        R = math.fabs(R_gen.gen())
        C = math.fabs(C_gen.gen())
        V_in = math.fabs(Vin_gen.gen())
        if q0_mean: q0 = math.fabs(q0_gen.gen())
        if i0_mean: i0 = math.fabs(i0_gen.gen())


        V_L_list.append(V_in / L)
        R_L_list.append(-R / L)
        rev_LC_list.append(-1 / (L * C))

        rlc = RLC_Circuit_Modified(L=L, R=R, C=C, V_in=V_in, t_start=t_span[0], t_end=t_span[1], dt=dt, noise_std=noise_std)#(L=L, R=R, C=C, V_in=V_in, q0=q0, i0=i0)
        dq_dt, di_dt = rlc.simulate()
        X = np.array([np.ones_like(rlc.results['Charge']),
                      rlc.results['Charge'],
                      rlc.results['Current'],
                      rlc.results['Charge'] * rlc.results['Current'],
                      rlc.results['Charge'] ** 2,
                      rlc.results['Current'] ** 2])
        Y = np.array([dq_dt, di_dt])
        # X contains [1, q, i] for SINDy-like regression
        # X = np.array([np.ones_like(results['Charge']), results['Charge'], results['Current'], results['Charge']*results['Current'],results['Charge']**2, results['Current']**2])
        # Y = np.array([results['dq_dt'], results['di_dt']])

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    real_params = {
        'V/L_mean': np.mean(np.array(V_L_list)),
        'V/L_std': np.std(np.array(V_L_list)),
        '-R/L_mean': np.mean(np.array(R_L_list)),
        '-R/L_std': np.std(np.array(R_L_list)),
        '-1/LC_mean': np.mean(np.array(rev_LC_list)),
        '-1/LC_std': np.std(np.array(rev_LC_list)),

        'V/L_array': np.array(V_L_list),
        '-R/L_array': np.array(R_L_list),
        '-1/LC_array': np.array(rev_LC_list),
    }


    return X_all, Y_all, real_params

def gt_utils(real_params):
    eqs = ["dq/dt = i", "di/dt = (V/L) + (-R/L) * i + (-1/LC) * q"]
    coef_names = ["constant", "q", "i", "q*i","q**2","i**2"] # Updated coef_names for RLC

    gt_coef = [{'constant': [0, 0], 'q': [0, 0], 'i': [1, 0],"q*i":[0,0],"q**2":[0,0],"i**2":[0,0]},
               {'constant': [real_params['V/L_mean'], real_params['V/L_std']],
                'q': [real_params['-1/LC_mean'], real_params['-1/LC_std']], # Swapped q and i coefficients to match equation
                'i': [real_params['-R/L_mean'], real_params['-R/L_std']],
                "q*i":[0,0],"q**2":[0,0],"i**2":[0,0]}]
    return {"eqs":eqs, "coef_names":coef_names,"gt_coef":gt_coef}

def realparame2gtarray(real_params:dict):
    # real_params = {
    #     'V/L_array': np.array(V_L_list),
    #     '-R/L_array': np.array(R_L_list),
    #     '-1/LC_array': np.array(rev_LC_list),
    # }
    # ["constant", "q", "i", "q*i", "q**2", "i**2"]
    dqdt_const_arr = np.zeros_like(real_params['V/L_array'])
    dqdt_q_arr = np.zeros_like(real_params['V/L_array'])
    dqdt_i_arr = np.ones_like(real_params['V/L_array'])
    dqdt_qi_arr = np.zeros_like(real_params['V/L_array'])
    dqdt_q2_arr = np.zeros_like(real_params['V/L_array'])
    dqdt_i2_arr = np.zeros_like(real_params['V/L_array'])

    didt_const_arr = real_params['V/L_array']
    didt_q_arr = real_params['-1/LC_array']
    didt_i_arr = real_params['-R/L_array']
    didt_qi_arr = np.zeros_like(real_params['V/L_array'])
    didt_q2_arr = np.zeros_like(real_params['V/L_array'])
    didt_i2_arr = np.zeros_like(real_params['V/L_array'])

    eq1_coef = [dqdt_const_arr, dqdt_q_arr, dqdt_i_arr, dqdt_qi_arr, dqdt_q2_arr, dqdt_i2_arr]
    eq2_coef = [didt_const_arr, didt_q_arr, didt_i_arr, didt_qi_arr, didt_q2_arr, didt_i2_arr]

    eqs = np.array([eq1_coef, eq2_coef])
    return eqs


def generate_pdf(save_path, pdf_smaple_N=10000, epsilon = 0.001 ):
    """
    Generates probability density function (PDF) samples for RLC circuit equation coefficients.

    Args:
        save_path (str): Path to the directory containing the system_param_dict.pkl file.
        pdf_smaple_N (int): Number of samples to generate for the PDF (default: 10000).
        epsilon (float): Small value for generating fixed parameter distributions (default: 0.001).

    Returns:
        numpy.ndarray: An array containing the PDF samples for the coefficients of the RLC circuit equations.
                       The shape is (num_equations, num_coefficients, pdf_sample_N).
    """
    with open(os.path.join(save_path, "system_param_dict.pkl"), "rb") as f:
        system_param_dict = pickle.load(f)

    L_V = system_param_dict['L_info']["L_V"] if "L_V" in list(system_param_dict["L_info"].keys()) else None
    L_N = system_param_dict['L_info']["L_N"] if "L_N" in list(system_param_dict["L_info"].keys()) else None
    L_mean = system_param_dict['L_info']["L_mean"] if "L_mean" in list(system_param_dict["L_info"].keys()) else None
    L_std = system_param_dict['L_info']["L_std"] if "L_std" in list(system_param_dict["L_info"].keys()) else None

    R_V = system_param_dict['R_info']["R_V"] if "R_V" in list(system_param_dict["R_info"].keys()) else None
    R_N = system_param_dict['R_info']["R_N"] if "R_N" in list(system_param_dict["R_info"].keys()) else None
    R_mean = system_param_dict['R_info']["R_mean"] if "R_mean" in list(system_param_dict["R_info"].keys()) else None
    R_std = system_param_dict['R_info']["R_std"] if "R_std" in list(system_param_dict["R_info"].keys()) else None

    C_V = system_param_dict['C_info']["C_V"] if "C_V" in list(system_param_dict["C_info"].keys()) else None
    C_N = system_param_dict['C_info']["C_N"] if "C_N" in list(system_param_dict["C_info"].keys()) else None
    C_mean = system_param_dict['C_info']["C_mean"] if "C_mean" in list(system_param_dict["C_info"].keys()) else None
    C_std = system_param_dict['C_info']["C_std"] if "C_std" in list(system_param_dict["C_info"].keys()) else None

    V_in_V = system_param_dict['V_in_info']["V_in_V"] if "V_in_V" in list(system_param_dict["V_in_info"].keys()) else None
    V_in_N = system_param_dict['V_in_info']["V_in_N"] if "V_in_N" in list(system_param_dict["V_in_info"].keys()) else None
    V_in_mean = system_param_dict['V_in_info']["V_in_mean"] if "V_in_mean" in list(
        system_param_dict["V_in_info"].keys()) else None
    V_in_std = system_param_dict['V_in_info']["V_in_std"] if "V_in_std" in list(system_param_dict["V_in_info"].keys()) else None


    if L_V is not None:
        L_vals = np.abs(np.random.normal(L_V,epsilon, pdf_smaple_N))
    else:
        L_vals = np.abs(np.random.normal(L_mean, L_std, pdf_smaple_N))

    if R_V is not None:
        R_vals = np.abs(np.random.normal(R_V,epsilon, pdf_smaple_N))
    else:
        R_vals = np.abs(np.random.normal(R_mean, R_std, pdf_smaple_N))

    if C_V is not None:
        C_vals = np.abs(np.random.normal(C_V,epsilon, pdf_smaple_N))
    else:
        C_vals = np.abs(np.random.normal(C_mean, C_std, pdf_smaple_N))

    if V_in_V is not None:
        V_in_vals = np.abs(np.random.normal(V_in_V,epsilon, pdf_smaple_N))
    else:
        V_in_vals = np.abs(np.random.normal(V_in_mean, V_in_std, pdf_smaple_N))


    # RLC circuit equations:
    # dq/dt = i
    # di/dt = (V_in/L) - (R/L) * i - (1/LC) * q
    # Coefficients are for terms: [constant, q, i, q*i, q**2, i**2]

    # Coefficients for dq/dt = i
    dqdt_const_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))
    dqdt_q_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))
    dqdt_i_arr = np.abs(np.random.normal(1, epsilon, pdf_smaple_N))
    dqdt_qi_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))
    dqdt_q2_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))
    dqdt_i2_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))

    # Coefficients for di/dt = (V_in/L) - (R/L) * i - (1/LC) * q
    # Ensure no division by zero for L and C
    L_vals[L_vals == 0] = epsilon
    C_vals[C_vals == 0] = epsilon

    didt_const_arr = V_in_vals / L_vals
    didt_q_arr = -1 / (L_vals * C_vals)
    didt_i_arr = -R_vals / L_vals
    didt_qi_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))
    didt_q2_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))
    didt_i2_arr = np.abs(np.random.normal(0, epsilon, pdf_smaple_N))

    coef_list_1 = [dqdt_const_arr, dqdt_q_arr, dqdt_i_arr, dqdt_qi_arr, dqdt_q2_arr, dqdt_i2_arr]
    coef_list_2 = [didt_const_arr, didt_q_arr, didt_i_arr, didt_qi_arr, didt_q2_arr, didt_i2_arr]


    pdf_list = [coef_list_1,coef_list_2]
    pdf_arr = np.array(pdf_list)
    return pdf_arr
