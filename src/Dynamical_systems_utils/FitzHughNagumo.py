import math
import pickle

from scipy.integrate import solve_ivp
import numpy as np
from src.utils import gen_param
import os


# os.environ["JAX_TRACEBACK_FILTERING"] = "off" # Keep this if you have JAX issues, otherwise it can be removed

class FitzHughNagumo:
    """
    A class to simulate the FitzHugh-Nagumo system.
    """

    def __init__(self, a, b0, b1, I, v0=0.0, w0=0., gradient_targets = True):
        """
        Initializes the FitzHugh-Nagumo parameters and initial conditions.

        Args:
            a (float): Parameter 'a' from the FHN equations.
            b0 (float): Parameter 'b0' from the FHN equations.
            b1 (float): Parameter 'b1' from the FHN equations.
            I (float): External current 'I'.
            v0 (float): Initial membrane potential (default: 0.0).
            w0 (float): Initial recovery variable (default: 0.0).
        """
        self.a = a
        self.b0 = b0
        self.b1 = b1
        self.I = I
        self.v0 = v0
        self.w0 = w0
        self.gradient_targets = gradient_targets

    def _equations(self, t, state):
        """
        Defines the equations of motion for the FitzHugh-Nagumo system.

        Args:
            t (float): Current time (not explicitly used in FHN equations, but required by solve_ivp).
            state (numpy.ndarray): A NumPy array containing the current membrane potential (v) and recovery variable (w).

        Returns:
            numpy.ndarray: A NumPy array containing the derivatives dv/dt and dw/dt.
        """
        v, w = state

        dvdt = v - (v ** 3) / 3 - w + self.I
        dwdt = self.a * self.b0 + self.a * self.b1 * v - self.a * w

        return [dvdt, dwdt]

    def simulate(self, t_span=(0, 10), dt=0.01, noise_std=0.01):
        """
        Simulates the FitzHugh-Nagumo system over a given time span.

        Args:
            t_span (tuple): A tuple (t_start, t_end) defining the time interval of simulation (default: (0, 10)).
            dt (float): The time step for generating the output time points (default: 0.01).
            noise_std (float): The standard deviation of the Gaussian noise to add to the simulated data (default: 0.01).

        Returns:
            dict: A dictionary containing the time points ('t'), membrane potential ('v'), recovery variable ('w'),
                  and their derivatives ('dv_dt', 'dw_dt') with added noise.
        """
        sol = solve_ivp(self._equations, t_span, [self.v0, self.w0],
                        dense_output=True, method='RK45')  # Using RK45 for accuracy
        t_eval = np.arange(t_span[0], t_span[1] , dt)
        results = sol.sol(t_eval)

        # Add noise to the state variables (v and w)
        v = results[0] + np.random.normal(0, noise_std, len(t_eval))
        w = results[1] + np.random.normal(0, noise_std, len(t_eval))

        # Calculate derivatives (without noise for the "true" Y, then add noise for output)
        # These are directly from the FHN equations using the 'true' simulated states
        if self.gradient_targets:
            # Compute numerical gradients
            dv_dt = np.gradient(v, dt)
            dw_dt = np.gradient(w, dt)
        else:
            dv_dt = results[0] - (results[0] ** 3) / 3 - results[1] + self.I
            dw_dt = self.a * self.b0 + self.a * self.b1 * results[0] - self.a * results[1]



        return {
            't': t_eval,
            'v': v,
            'w': w,
            'dv_dt': dv_dt,
            'dw_dt': dw_dt
        }


print("---------------------------- Mix Function -------------------------")


def mix_data(system_param_dict):
    N_param_set = system_param_dict['N_param_set']
    a_V = system_param_dict['a_info']["a_V"] if "a_V" in list(system_param_dict["a_info"].keys()) else None
    a_mean = system_param_dict['a_info']["a_mean"] if "a_mean" in list(system_param_dict["a_info"].keys()) else None
    a_std = system_param_dict['a_info']["a_std"] if "a_std" in list(system_param_dict["a_info"].keys()) else None

    b0_V = system_param_dict['b0_info']["b0_V"] if "b0_V" in list(system_param_dict["b0_info"].keys()) else None
    b0_mean = system_param_dict['b0_info']["b0_mean"] if "b0_mean" in list(
        system_param_dict["b0_info"].keys()) else None
    b0_std = system_param_dict['b0_info']["b0_std"] if "b0_std" in list(system_param_dict["b0_info"].keys()) else None

    b1_V = system_param_dict['b1_info']["b1_V"] if "b1_V" in list(system_param_dict["b1_info"].keys()) else None
    b1_mean = system_param_dict['b1_info']["b1_mean"] if "b1_mean" in list(
        system_param_dict["b1_info"].keys()) else None
    b1_std = system_param_dict['b1_info']["b1_std"] if "b1_std" in list(system_param_dict["b1_info"].keys()) else None

    I_V = system_param_dict['I_info']["I_V"] if "I_V" in list(system_param_dict["I_info"].keys()) else None
    I_mean = system_param_dict['I_info']["I_mean"] if "I_mean" in list(
        system_param_dict["I_info"].keys()) else None
    I_std = system_param_dict['I_info']["I_std"] if "I_std" in list(system_param_dict["I_info"].keys()) else None

    v0_V = system_param_dict['v0_info']["v0_V"] if "v0_V" in list(system_param_dict["v0_info"].keys()) else None
    v0_mean = system_param_dict['v0_info']["v0_mean"] if "v0_mean" in list(
        system_param_dict["v0_info"].keys()) else None
    v0_std = system_param_dict['v0_info']["v0_std"] if "v0_std" in list(system_param_dict["v0_info"].keys()) else None

    w0_V = system_param_dict['w0_info']["w0_V"] if "w0_V" in list(system_param_dict["w0_info"].keys()) else None
    w0_mean = system_param_dict['w0_info']["w0_mean"] if "w0_mean" in list(
        system_param_dict["w0_info"].keys()) else None
    w0_std = system_param_dict['w0_info']["w0_std"] if "w0_std" in list(system_param_dict["w0_info"].keys()) else None

    t_start = system_param_dict['t_info']["t_start"] if "t_start" in list(system_param_dict["t_info"].keys()) else 0
    t_end = system_param_dict['t_info']["t_end"] if "t_end" in list(system_param_dict["t_info"].keys()) else 10
    dt = system_param_dict['t_info']["dt"] if "dt" in list(system_param_dict["t_info"].keys()) else 0.01
    noise_std = system_param_dict['noise_info']["noise_std"] if "noise_std" in list(
        system_param_dict["noise_info"].keys()) else 0.01

    # # Generate parameter values based on fixed value or distribution
    # if a_V is not None:
    #     a_vals = np.array([a_V])
    # else:
    #     a_vals = np.abs(np.random.normal(a_mean, a_std, a_N))
    #
    # if b0_V is not None:
    #     b0_vals = np.array([b0_V])
    # else:
    #     b0_vals = np.abs(np.random.normal(b0_mean, b0_std, b0_N))
    #
    # if b1_V is not None:
    #     b1_vals = np.array([b1_V])
    # else:
    #     b1_vals = np.abs(np.random.normal(b1_mean, b1_std, b1_N))
    #
    # if I_V is not None:
    #     I_vals = np.array([I_V])
    # else:
    #     I_vals = np.abs(np.random.normal(I_mean, I_std, I_N))
    #
    # if v0_V is not None:
    #     v0_vals = np.array([v0_V])
    # else:
    #     v0_vals = np.random.normal(v0_mean, v0_std, v0_N)  # v0 can be negative
    #
    # if w0_V is not None:
    #     w0_vals = np.array([w0_V])
    # else:
    #     w0_vals = np.random.normal(w0_mean, w0_std, w0_N)  # w0 can be negative
    #
    # param_sets = []
    # # Loop through all combinations of parameters
    # for a in a_vals:
    #     for b0 in b0_vals:
    #         for b1 in b1_vals:
    #             for I in I_vals:
    #                 for v0 in v0_vals:
    #                     for w0 in w0_vals:
    #                         param_sets.append({
    #                             'a': a,
    #                             'b0': b0,
    #                             'b1': b1,
    #                             'I': I,
    #                             'v0': v0,
    #                             'w0': w0
    #                         })

    X_all = []
    Y_all = []

    # Lists to store the 'true' coefficients for ground truth comparison
    # These correspond to the coefficients of the basis functions in the FHN equations
    v_coef_list = []
    v3_coef_list = []
    w_coef_dvdt_list = []
    I_coef_list = []

    constant_dwdt_list = []
    v_dwdt_list = []
    w_dwdt_list = []

    a_gen = gen_param(N_param=N_param_set,a_V=a_V,a_mean=a_mean,a_std=a_std)
    b0_gen = gen_param(N_param=N_param_set,a_V=b0_V,a_mean=b0_mean,a_std=b0_std)
    b1_gen = gen_param(N_param=N_param_set, a_V=b1_V, a_mean=b1_mean, a_std=b1_std)
    I_gen = gen_param(N_param=N_param_set, a_V=I_V, a_mean=I_mean, a_std=I_std)
    v0_gen = gen_param(N_param=N_param_set, a_V=v0_V, a_mean=v0_mean, a_std=v0_std)
    w0_gen = gen_param(N_param=N_param_set, a_V=w0_V, a_mean=w0_mean, a_std=w0_std)

    for param_i in range(N_param_set):
        a = math.fabs(a_gen.gen())
        b0 = math.fabs(b0_gen.gen())
        b1 = math.fabs(b1_gen.gen())
        I = math.fabs(I_gen.gen())
        v0 = math.fabs(v0_gen.gen())
        w0 = math.fabs(w0_gen.gen())

        # Store the 'true' coefficients for later comparison with identified models
        v_coef_list.append(1.0)  # Coefficient of v in dv/dt = v - v^3/3 - w + I
        v3_coef_list.append(-1 / 3.0)  # Coefficient of -v^3/3
        w_coef_dvdt_list.append(-1.0)  # Coefficient of w in dv/dt
        I_coef_list.append(I)  # Coefficient of I (constant term for dv/dt)

        constant_dwdt_list.append(a * b0)  # Constant term for dw/dt
        v_dwdt_list.append(a * b1)  # Coefficient of v in dw/dt
        w_dwdt_list.append(-a)  # Coefficient of w in dw/dt
        print(f"(a={a}, b0={b0}, b1={b1}, I={I}, v0={v0}, w0={w0})")
        fhn_system = FitzHughNagumo(a=a, b0=b0, b1=b1, I=I, v0=v0, w0=w0)
        results = fhn_system.simulate(t_span=(t_start, t_end), dt=dt, noise_std=noise_std)

        t = results['t']
        v = results['v']
        w = results['w']
        dv_dt = results['dv_dt']
        dw_dt = results['dw_dt']

        # X contains basis functions relevant for FHN:
        # [1, v, w, v^3]
        # dx/dt = c0*1 + c1*v + c2*w + c3*v^3
        # dv/dt = c4*1 + c5*v + c6*w + c7*v^3
        # Note: '1' term is for 'I' in dv/dt and 'a*b0' in dw/dt

        # Basis functions for the FHN equations
        X = np.array([
            np.ones_like(v),  # Constant term
            v,  # v
            w,  # w
            v ** 3,  # v^3
            v ** 2,  # v^2 (often included for general polynomial library)
            w ** 2,  # w^2
            v * w  # vw
        ])

        Y = np.array([dv_dt, dw_dt])  # Derivatives

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    # Store statistics of the 'true' coefficients for ground truth comparison
    real_params = {
        'dv/dt_constant_mean': np.mean(np.array(I_coef_list)),  # I
        'dv/dt_constant_std': np.std(np.array(I_coef_list)),

        'dv/dt_v_mean': np.mean(np.array(v_coef_list)),  # 1.0
        'dv/dt_v_std': np.std(np.array(v_coef_list)),

        'dv/dt_v3_mean': np.mean(np.array(v3_coef_list)),  # -1/3.0
        'dv/dt_v3_std': np.std(np.array(v3_coef_list)),

        'dv/dt_w_mean': np.mean(np.array(w_coef_dvdt_list)),  # -1.0
        'dv/dt_w_std': np.std(np.array(w_coef_dvdt_list)),

        'dw/dt_constant_mean': np.mean(np.array(constant_dwdt_list)),  # a*b0
        'dw/dt_constant_std': np.std(np.array(constant_dwdt_list)),

        'dw/dt_v_mean': np.mean(np.array(v_dwdt_list)),  # a*b1
        'dw/dt_v_std': np.std(np.array(v_dwdt_list)),

        'dw/dt_w_mean': np.mean(np.array(w_dwdt_list)),  # -a
        'dw/dt_w_std': np.std(np.array(w_dwdt_list)),

        # Also store arrays of the true coefficients if needed for more detailed analysis
        'dv/dt_constant_array': np.array(I_coef_list),
        'dv/dt_v_array': np.array(v_coef_list),
        'dv/dt_v3_array': np.array(v3_coef_list),
        'dv/dt_w_array': np.array(w_coef_dvdt_list),

        'dw/dt_constant_array': np.array(constant_dwdt_list),
        'dw/dt_v_array': np.array(v_dwdt_list),
        'dw/dt_w_array': np.array(w_dwdt_list)
    }

    return X_all, Y_all, real_params


def gt_utils(real_params):
    eqs = ["dv/dt = I + v - (1/3)v^3 - w", "dw/dt = a*b0 + a*b1*v - a*w"]
    # Adjust coef_names to match the X array created in mix_data
    coef_names = ["constant", "v", "w", "v**3", "v**2", "w**2", "v*w"]

    # Ground truth coefficients for the dv/dt equation
    # [constant (I), v, w, v^3, v^2, w^2, v*w]
    gt_coef_dv = {
        'constant': [real_params['dv/dt_constant_mean'], real_params['dv/dt_constant_std']],
        'v': [real_params['dv/dt_v_mean'], real_params['dv/dt_v_std']],
        'w': [real_params['dv/dt_w_mean'], real_params['dv/dt_w_std']],
        'v**3': [real_params['dv/dt_v3_mean'], real_params['dv/dt_v3_std']],
        'v**2': [0, 0],  # No v^2 term in FHN
        'w**2': [0, 0],  # No w^2 term in FHN
        'v*w': [0, 0]  # No v*w term in FHN
    }

    # Ground truth coefficients for the dw/dt equation
    # [constant (a*b0), v (a*b1), w (-a), v^3, v^2, w^2, v*w]
    gt_coef_dw = {
        'constant': [real_params['dw/dt_constant_mean'], real_params['dw/dt_constant_std']],
        'v': [real_params['dw/dt_v_mean'], real_params['dw/dt_v_std']],
        'w': [real_params['dw/dt_w_mean'], real_params['dw/dt_w_std']],
        'v**3': [0, 0],  # No v^3 term in FHN dw/dt
        'v**2': [0, 0],  # No v^2 term in FHN dw/dt
        'w**2': [0, 0],  # No w^2 term in FHN dw/dt
        'v*w': [0, 0]  # No v*w term in FHN dw/dt
    }

    # Combine into a list for dv/dt and dw/dt
    gt_coef = [gt_coef_dv, gt_coef_dw]
    eq_list = []
    for eq in gt_coef:
        coef_list = []
        for k, v in eq.items():
            coef_list += [v]
        eq_list+=[coef_list]
    gt_info_arr = np.array(eq_list)
    return {"eqs": eqs, "coef_names": coef_names, "gt_coef": gt_coef, "gt_info_arr": gt_info_arr}


def realparame2gtarray(real_params:dict):

    dvdt_const_arr = real_params['dv/dt_constant_array']
    dvdt_v_arr = real_params['dv/dt_v_array']
    dvdt_w_arr = real_params['dv/dt_w_array']
    dvdt_v3_arr = real_params['dv/dt_v3_array']
    dvdt_v2_arr = np.zeros_like(dvdt_v3_arr)
    dvdt_w2_arr = np.zeros_like(dvdt_v3_arr)
    dvdt_vw_arr = np.zeros_like(dvdt_v3_arr)


    dwdt_const_arr = real_params['dw/dt_constant_array']
    dwdt_v_arr = real_params['dw/dt_v_array']
    dwdt_w_arr = real_params['dw/dt_w_array']
    dwdt_v3_arr = np.zeros_like(dvdt_v3_arr)
    dwdt_v2_arr = np.zeros_like(dvdt_v3_arr)
    dwdt_w2_arr = np.zeros_like(dvdt_v3_arr)
    dwdt_vw_arr = np.zeros_like(dvdt_v3_arr)

    # Correcting concatenation syntax
    gt_arr1 = np.array([
        dvdt_const_arr, dvdt_v_arr, dvdt_w_arr, dvdt_v3_arr, dvdt_v2_arr, dvdt_w2_arr, dvdt_vw_arr])
    gt_arr2 = np.array([
        dwdt_const_arr, dwdt_v_arr, dwdt_w_arr, dwdt_v3_arr, dwdt_v2_arr, dwdt_w2_arr, dwdt_vw_arr
    ]) # Concatenate the arrays
    gt_arr = np.array([gt_arr1, gt_arr2]) # Concatenate the arrays

    return gt_arr

def generate_pdf(save_path, pdf_smaple_N=10000, epsilon = 0.01):
    """

    :param mean_std_arr:
    it is an array with shape :(N_eq, N_coef, 2), means that for example the element [2,3,:]
    contains the mean and std of the 3 coef in the 2nd equation. Then [2,3,0] is mean and [2,3,1] is the std.
    The imposed_sgn is the sign of each coef in the ground truth equation.
    :return:
    the output of this function is an array of the form (N_eq,N_coef, pdf_smaple_N).
    Depending on the dynamical system, sometimes the pdf comes from a normal dist, somthimes comes from a abs of
    normal dist, and somethimes comes from a truncated dist. Thast why it should be defined as a function for each
    dynamical system.
    eqs = ["dv/dt = I + v - (1/3)v^3 - w", "dw/dt = a*b0 + a*b1*v - a*w"]
    coef_names = ["constant", "v", "w", "v**3", "v**2", "w**2", "v*w"]

    """
    with open(os.path.join(save_path,"system_param_dict.pkl"),"rb") as f:
        system_param_dict = pickle.load(f)


    a_V = system_param_dict['a_info']["a_V"] if "a_V" in list(system_param_dict["a_info"].keys()) else None
    a_mean = system_param_dict['a_info']["a_mean"] if "a_mean" in list(system_param_dict["a_info"].keys()) else None
    a_std = system_param_dict['a_info']["a_std"] if "a_std" in list(system_param_dict["a_info"].keys()) else None

    b0_V = system_param_dict['b0_info']["b0_V"] if "b0_V" in list(system_param_dict["b0_info"].keys()) else None
    b0_mean = system_param_dict['b0_info']["b0_mean"] if "b0_mean" in list(
        system_param_dict["b0_info"].keys()) else None
    b0_std = system_param_dict['b0_info']["b0_std"] if "b0_std" in list(system_param_dict["b0_info"].keys()) else None

    b1_V = system_param_dict['b1_info']["b1_V"] if "b1_V" in list(system_param_dict["b1_info"].keys()) else None
    b1_mean = system_param_dict['b1_info']["b1_mean"] if "b1_mean" in list(
        system_param_dict["b1_info"].keys()) else None
    b1_std = system_param_dict['b1_info']["b1_std"] if "b1_std" in list(system_param_dict["b1_info"].keys()) else None

    I_V = system_param_dict['I_info']["I_V"] if "I_V" in list(system_param_dict["I_info"].keys()) else None
    I_mean = system_param_dict['I_info']["I_mean"] if "I_mean" in list(
        system_param_dict["I_info"].keys()) else None
    I_std = system_param_dict['I_info']["I_std"] if "I_std" in list(system_param_dict["I_info"].keys()) else None





    I_list = []

    ab0_list = []
    ab1_list = []
    a_list = []
    constant_dwdt_list = []

    a_gen = gen_param(pdf_smaple_N, a_V, a_mean, a_std)
    b0_gen = gen_param(pdf_smaple_N, b0_V, b0_mean, b0_std)
    b1_gen = gen_param(pdf_smaple_N, b1_V, b1_mean, b1_std)
    I_gen = gen_param(pdf_smaple_N, I_V, I_mean, I_std)

    for param_i in range(pdf_smaple_N):
        a = math.fabs(a_gen.gen())
        b0 = math.fabs(b0_gen.gen())
        b1 = math.fabs(b1_gen.gen())
        I = math.fabs(b1_gen.gen())
        I_list.append(I)
        ab0_list.append(a*b0)
        ab1_list.append(a*b1)
        a_list.append(a)

        # eqs = ["dv/dt = I + v - (1/3)v^3 - w", "dw/dt = a*b0 + a*b1*v - a*w"]
        # # Adjust coef_names to match the X array created in mix_data
        # coef_names = ["constant", "v", "w", "v**3", "v**2", "w**2", "v*w"]

    coef_list_0 = [np.array(I_list),  # const
                  np.abs(np.random.normal(1, epsilon, pdf_smaple_N)),  #  #
                  np.abs(np.random.normal(-1, epsilon, pdf_smaple_N)),
                  np.abs(np.random.normal(-1 / 3., epsilon, pdf_smaple_N)), #
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),  #
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N))]

    coef_list_1 = [np.array(ab0_list),  # const
                  np.array(ab1_list),  #  #
                  -1*np.array(a_list),
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N)), #
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),  #
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
                  np.abs(np.random.normal(0, epsilon, pdf_smaple_N))]

    pdf_arr = np.array([coef_list_0,coef_list_1])
    return pdf_arr
