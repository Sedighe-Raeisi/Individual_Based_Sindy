from scipy.integrate import solve_ivp
import numpy as np
import os
import pickle
import math
from src.pdf_utils import gen_param
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
class HarmonicOscillator:
    """
    A class to simulate a simple harmonic oscillator system.
    """
    def __init__(self, m, k, c=0.0, F0=0.0, omega=0.0, x0=1.0, v0=0.0, gradient_targets = True):
        """
        Initializes the harmonic oscillator parameters and initial conditions.

        Args:
            m (float): Mass of the object.
            k (float): Spring constant.
            c (float): Damping coefficient (default: 0.0 for no damping).
            F0 (float): Amplitude of the forcing term (default: 0.0 for no forcing).
            omega (float): Angular frequency of the forcing term (default: 0.0 for no forcing).
            x0 (float): Initial position (default: 1.0).
            v0 (float): Initial velocity (default: 0.0).
        """
        self.m = m
        self.k = k
        self.c = c
        self.F0 = F0
        self.omega = omega
        self.x0 = x0
        self.v0 = v0
        self.gradient_targets = gradient_targets

    def _equations(self, t, state):
        """
        Defines the equations of motion for the forced, damped harmonic oscillator.

        Args:
            t (float): Current time.
            state (numpy.ndarray): A NumPy array containing the current position (x) and velocity (v).

        Returns:
            numpy.ndarray: A NumPy array containing the derivatives dx/dt (velocity) and dv/dt (acceleration).
        """
        x, v = state
        dxdt = v
        dvdt = -1*(self.k / self.m ) * x - (self.c / self.m) * v + (self.F0 / self.m) * np.cos(self.omega * t)
        return [dxdt, dvdt]

    def simulate(self, t_span=(0, 10), dt=0.01, noise_level=0.01):
        """
        Simulates the forced, damped harmonic oscillator over a given time span.

        Args:
            t_span (tuple): A tuple (t_start, t_end) defining the time interval of simulation (default: (0, 10)).
            dt (float): The time step for generating the output time points (default: 0.01).
            noise_std (float): The standard deviation of the Gaussian noise to add to the simulated data (default: 0.01).

        Returns:
            dict: A dictionary containing the time points ('t'), position ('x'), velocity ('v'),
                  and their derivatives ('dx_dt', 'dv_dt') with added noise.
        """
        sol = solve_ivp(self._equations, t_span, [self.x0, self.v0],
                        dense_output=True, method='RK45')  # Using RK45 for accuracy
        t_eval = np.arange(t_span[0], t_span[1] , dt)
        results = sol.sol(t_eval)

        x = results[0]
        v = results[1]

        if self.gradient_targets:
            # Compute numerical gradients
            dx_dt = np.gradient(x, dt)
            dv_dt = np.gradient(v, dt)
        else:
            dx_dt = results[1]
            dv_dt = -self.k / self.m * results[0] - self.c / self.m * results[1] + self.F0 / self.m * np.cos(
                self.omega * t_eval)

        xnoise_scale = np.std(results[0]) * noise_level
        vnoise_scale = np.std(results[1]) * noise_level

        x = results[0] + np.random.normal(0, xnoise_scale, len(t_eval))
        v = results[1] + np.random.normal(0, vnoise_scale, len(t_eval))


        return {
            't': t_eval,
            'x': x,
            'v': v,
            'dx_dt': dx_dt,
            'dv_dt': dv_dt
        }
print("---------------------------- Mix Function -------------------------")
def mix_data(system_param_dict):

    """
    Generates mixed data for the forced, damped harmonic oscillator system with varying parameters.

    Args:
        m_V (float, optional): Fixed value for mass. Defaults to None.
        m_N (int, optional): Number of mass samples if m_V is None. Defaults to None.
        m_mean (float, optional): Mean of mass distribution if m_V is None. Defaults to None.
        m_std (float, optional): Standard deviation of mass distribution if m_V is None. Defaults to None.
        k_V (float, optional): Fixed value for spring constant. Defaults to None.
        k_N (int, optional): Number of spring constant samples if k_V is None. Defaults to None.
        k_mean (float, optional): Mean of spring constant distribution if k_V is None. Defaults to None.
        k_std (float, optional): Standard deviation of spring constant distribution if k_V is None. Defaults to None.
        c_V (float, optional): Fixed value for damping coefficient. Defaults to None.
        c_N (int, optional): Number of damping coefficient samples if c_V is None. Defaults to None.
        c_mean (float, optional): Mean of damping coefficient distribution if c_V is None. Defaults to None.
        c_std (float, optional): Standard deviation of damping coefficient distribution if c_V is None. Defaults to None.
        F0_V (float, optional): Fixed value for forcing amplitude. Defaults to None.
        F0_N (int, optional): Number of forcing amplitude samples if F0_V is None. Defaults to None.
        F0_mean (float, optional): Mean of forcing amplitude distribution if F0_V is None. Defaults to None.
        F0_std (float, optional): Standard deviation of forcing amplitude distribution if F0_V is None. Defaults to None.
        omega_V (float, optional): Fixed value for forcing frequency. Defaults to None.
        omega_N (int, optional): Number of forcing frequency samples if omega_V is None. Defaults to None.
        omega_mean (float, optional): Mean of forcing frequency distribution if omega_V is None. Defaults to None.
        omega_std (float, optional): Standard deviation of forcing frequency distribution if omega_V is None. Defaults to None.
        t_start (int, optional): Start time for simulation. Defaults to 0.
        t_end (int, optional): End time for simulation. Defaults to 10.
        dt (float, optional): Time step for simulation output. Defaults to 0.01.
        noise_level (float, optional): noise_level of noise added to data. Defaults to 0.01.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - X_all (numpy.ndarray): Array of input features [1, x, v, cos(omega*t)] for each time step and parameter set.
               - Y_all (numpy.ndarray): Array of target derivatives [dx/dt, dv/dt] for each time step and parameter set.
               - real_params (dict): Dictionary containing the mean and std of the generated parameters.
    """
    N_param_set = system_param_dict['N_param_set']
    m_V = system_param_dict['m_info']["m_V"] if "m_V" in list(system_param_dict["m_info"].keys()) else None
    m_mean = system_param_dict['m_info']["m_mean"] if "m_mean" in list(system_param_dict["m_info"].keys()) else None
    m_std = system_param_dict['m_info']["m_std"] if "m_std" in list(system_param_dict["m_info"].keys()) else None

    k_V = system_param_dict['k_info']["k_V"] if "k_V" in list(system_param_dict["k_info"].keys()) else None
    k_mean = system_param_dict['k_info']["k_mean"] if "k_mean" in list(system_param_dict["k_info"].keys()) else None
    k_std = system_param_dict['k_info']["k_std"] if "k_std" in list(system_param_dict["k_info"].keys()) else None

    c_V = system_param_dict['c_info']["c_V"] if "c_V" in list(system_param_dict["c_info"].keys()) else None
    c_mean = system_param_dict['c_info']["c_mean"] if "c_mean" in list(system_param_dict["c_info"].keys()) else None
    c_std = system_param_dict['c_info']["c_std"] if "c_std" in list(system_param_dict["c_info"].keys()) else None

    F0_V = system_param_dict['F0_info']["F0_V"] if "F0_V" in list(system_param_dict["F0_info"].keys()) else None
    F0_mean = system_param_dict['F0_info']["F0_mean"] if "F0_mean" in list(
        system_param_dict["F0_info"].keys()) else None
    F0_std = system_param_dict['F0_info']["F0_std"] if "F0_std" in list(system_param_dict["F0_info"].keys()) else None
    F_zero_portion = system_param_dict['F0_info']["F_zero_portion"] if "F_zero_portion" in list(
        system_param_dict["F0_info"].keys()) else None

    omega_V = system_param_dict['omega_info']["omega_V"] if "omega_V" in list(
        system_param_dict["omega_info"].keys()) else None
    omega_mean = system_param_dict['omega_info']["omega_mean"] if "omega_mean" in list(
        system_param_dict["omega_info"].keys()) else None
    omega_std = system_param_dict['omega_info']["omega_std"] if "omega_std" in list(
        system_param_dict["omega_info"].keys()) else None

    x0_V = system_param_dict['x0_info']["x0_V"] if "x0_V" in list(system_param_dict["x0_info"].keys()) else None
    x0_mean = system_param_dict['x0_info']["x0_mean"] if "x0_mean" in list(
        system_param_dict["x0_info"].keys()) else None
    x0_std = system_param_dict['x0_info']["x0_std"] if "x0_std" in list(system_param_dict["x0_info"].keys()) else None

    v0_V = system_param_dict['v0_info']["v0_V"] if "v0_V" in list(system_param_dict["v0_info"].keys()) else None
    v0_mean = system_param_dict['v0_info']["v0_mean"] if "v0_mean" in list(
        system_param_dict["v0_info"].keys()) else None


    t_start = system_param_dict['t_info']["t_start"] if "t_start" in list(system_param_dict["t_info"].keys()) else 0
    t_end = system_param_dict['t_info']["t_end"] if "t_end" in list(system_param_dict["t_info"].keys()) else 10
    dt = system_param_dict['t_info']["dt"] if "dt" in list(system_param_dict["t_info"].keys()) else 0.01
    noise_level = system_param_dict['noise_info']["noise_level"] if "noise_level" in list(
        system_param_dict["noise_info"].keys()) else 0.01



    X_all = []
    Y_all = []

    k_m_list = []
    c_m_list = []
    F0_m_list = []

    m_gen = gen_param(N_param_set,m_V,m_mean,m_std)
    k_gen = gen_param(N_param_set,k_V,k_mean,k_std)
    c_gen = gen_param(N_param_set,c_V,c_mean,c_std)
    F0_gen = gen_param(N_param_set,F0_V,F0_mean,F0_std,F_zero_portion)
    omega_gen = gen_param(N_param_set, omega_V, omega_mean, omega_std)
    for param_i in range(N_param_set):
        m = math.fabs(m_gen.gen())
        k = math.fabs(k_gen.gen())
        c = math.fabs(c_gen.gen())
        F0 = math.fabs(F0_gen.gen())
        omega = math.fabs(omega_gen.gen())


        k_m_list.append(-k / m)
        c_m_list.append(-c/m)
        F0_m_list.append(F0/m)

        oscillator = HarmonicOscillator(m=m, k=k, c=c, F0=F0, omega=omega)
        results = oscillator.simulate(t_span=(t_start, t_end), dt=dt, noise_level=noise_level)

        t = results['t']
        x = results['x']
        v = results['v']
        dx_dt = results['dx_dt']
        dv_dt = results['dv_dt']

        # X contains [1, x, v, cos(omega*t), x**2, v**2, x*v] for SINDy-like regression
        X = np.array([np.ones_like(x), x, v, np.cos(omega * t), x**2, v**2, x*v])
        Y = np.array([dx_dt, dv_dt])

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    real_params = {
        '-k/m_mean': np.mean(np.array(k_m_list)),
        '-k/m_std': np.std(np.array(k_m_list)),
        '-c/m_mean': np.mean(np.array(c_m_list)),
        '-c/m_std': np.std(np.array(c_m_list)),
        'F0/m_mean': np.mean(np.array(F0_m_list)),
        'F0/m_std': np.std(np.array(F0_m_list)),

        '-k/m_array': np.array(k_m_list),
        '-c/m_array': np.array(c_m_list),
        'F0/m_array': np.array(F0_m_list),
    }


    return X_all, Y_all, real_params

def gt_utils(real_params):
    eqs = ["dx/dt = v", "dv/dt = -k/m * x - c/m * v + F0/m * cos(omega*t)"]
    coef_names = ["constant", "x", "v", "cos(omega*t)", "x**2", "v**2", "x*v"]

    gt_coef_dx = {'constant': [0, 0],
                  'x': [0, 0],
                  'v': [1, 0],
                  'cos(omega*t)': [0, 0],
                  'x**2': [0, 0],
                  'v**2': [0, 0],
                  'x*v': [0, 0]}

    # Ground truth coefficients for the dw/dt equation
    # [constant (a*b0), v (a*b1), w (-a), v^3, v^2, w^2, v*w]
    gt_coef_dv = {'constant': [0, 0],
                  'x': [real_params['-k/m_mean'], real_params['-k/m_std']],
                'v': [real_params['-c/m_mean'], real_params['-c/m_std']],
                'cos(omega*t)': [real_params['F0/m_mean'], real_params['F0/m_std']],
                  'x**2': [0, 0],
                  'v**2': [0, 0],
                  'x*v': [0, 0]
                  }

    # Combine into a list for dv/dt and dw/dt
    gt_coef = [gt_coef_dx, gt_coef_dv]
    eq_list = []
    for eq in gt_coef:
        coef_list = []
        for k, v in eq.items():
            coef_list += [v]
        eq_list += [coef_list]
    gt_info_arr = np.array(eq_list)
    return {"eqs": eqs, "coef_names": coef_names, "gt_coef": gt_coef, "gt_info_arr": gt_info_arr}


def realparame2gtarray(real_params:dict):
    # X contains [1, x, v, cos(omega*t), x**2, v**2, x*v] for SINDy-like regression
    # eqs = ["dx/dt = v", "dv/dt = -k/m * x - c/m * v + F0/m * cos(omega*t)"]
    dxdt_const_arr = np.zeros_like(real_params['-k/m_array'])
    dxdt_x_arr = np.zeros_like(real_params['-k/m_array'])
    dxdt_v_arr = np.ones_like(real_params['-k/m_array'])
    dxdt_cos_arr = np.zeros_like(real_params['-k/m_array'])
    dxdt_x2_arr = np.zeros_like(real_params['-k/m_array'])
    dxdt_v2_arr = np.zeros_like(real_params['-k/m_array'])
    dxdt_xv_arr = np.zeros_like(real_params['-k/m_array'])


    dvdt_const_arr = np.zeros_like(real_params['-k/m_array'])
    dvdt_x_arr = real_params['-k/m_array']
    dvdt_v_arr = real_params['-c/m_array']
    dvdt_cos_arr = real_params['F0/m_array']
    dvdt_x2_arr = np.zeros_like(real_params['-k/m_array'])
    dvdt_v2_arr = np.zeros_like(real_params['-k/m_array'])
    dvdt_xv_arr = np.zeros_like(real_params['-k/m_array'])

    # Correcting concatenation syntax
    gt_arr1 = np.array([
        dxdt_const_arr, dxdt_x_arr, dxdt_v_arr, dxdt_cos_arr, dxdt_x2_arr, dxdt_v2_arr, dxdt_xv_arr])
    gt_arr2 = np.array([
        dvdt_const_arr, dvdt_x_arr, dvdt_v_arr, dvdt_cos_arr, dvdt_x2_arr, dvdt_v2_arr, dvdt_xv_arr])

    gt_arr = np.array([gt_arr1, gt_arr2]) # Concatenate the arrays

    return gt_arr

def generate_pdf(save_path, imposed_sign =np.array([[1,1,-1,-1,1,1,1],[1,1,-1,1,1,1,1]]), pdf_smaple_N=10000, epsilon = 0.001 ):
    with open(os.path.join(save_path, "system_param_dict.pkl"), "rb") as f:
        system_param_dict = pickle.load(f)
    m_V = system_param_dict['m_info']["m_V"] if "m_V" in list(system_param_dict["m_info"].keys()) else None
    m_N = system_param_dict['m_info']["m_N"] if "m_N" in list(system_param_dict["m_info"].keys()) else None
    m_mean = system_param_dict['m_info']["m_mean"] if "m_mean" in list(system_param_dict["m_info"].keys()) else None
    m_std = system_param_dict['m_info']["m_std"] if "m_std" in list(system_param_dict["m_info"].keys()) else None

    k_V = system_param_dict['k_info']["k_V"] if "k_V" in list(system_param_dict["k_info"].keys()) else None
    k_N = system_param_dict['k_info']["k_N"] if "k_N" in list(system_param_dict["k_info"].keys()) else None
    k_mean = system_param_dict['k_info']["k_mean"] if "k_mean" in list(system_param_dict["k_info"].keys()) else None
    k_std = system_param_dict['k_info']["k_std"] if "k_std" in list(system_param_dict["k_info"].keys()) else None

    c_V = system_param_dict['c_info']["c_V"] if "c_V" in list(system_param_dict["c_info"].keys()) else None
    c_N = system_param_dict['c_info']["c_N"] if "c_N" in list(system_param_dict["c_info"].keys()) else None
    c_mean = system_param_dict['c_info']["c_mean"] if "c_mean" in list(system_param_dict["c_info"].keys()) else None
    c_std = system_param_dict['c_info']["c_std"] if "c_std" in list(system_param_dict["c_info"].keys()) else None

    F0_V = system_param_dict['F0_info']["F0_V"] if "F0_V" in list(system_param_dict["F0_info"].keys()) else None
    F0_N = system_param_dict['F0_info']["F0_N"] if "F0_N" in list(system_param_dict["F0_info"].keys()) else None
    F0_mean = system_param_dict['F0_info']["F0_mean"] if "F0_mean" in list(
        system_param_dict["F0_info"].keys()) else None
    F0_std = system_param_dict['F0_info']["F0_std"] if "F0_std" in list(system_param_dict["F0_info"].keys()) else None
    F_zero_portion = system_param_dict['F0_info']["F_zero_portion"] if "F_zero_portion" in list(
        system_param_dict["F0_info"].keys()) else None

    omega_V = system_param_dict['omega_info']["omega_V"] if "omega_V" in list(
        system_param_dict["omega_info"].keys()) else None
    omega_N = system_param_dict['omega_info']["omega_N"] if "omega_N" in list(
        system_param_dict["omega_info"].keys()) else None
    omega_mean = system_param_dict['omega_info']["omega_mean"] if "omega_mean" in list(
        system_param_dict["omega_info"].keys()) else None
    omega_std = system_param_dict['omega_info']["omega_std"] if "omega_std" in list(
        system_param_dict["omega_info"].keys()) else None


    if m_V is not None:
        m_vals = np.abs(np.random.normal(m_V,epsilon, pdf_smaple_N))
    else:
        m_vals = np.abs(np.random.normal(m_mean, m_std, pdf_smaple_N))

    if k_V is not None:
        k_vals = np.abs(np.random.normal(k_V,epsilon, pdf_smaple_N))
    else:
        k_vals = np.abs(np.random.normal(k_mean, k_std, pdf_smaple_N))

    if c_V is not None:
        c_vals = np.abs(np.random.normal(c_V,epsilon, pdf_smaple_N))
    else:
        c_vals = np.abs(np.random.normal(c_mean, c_std, pdf_smaple_N))

    if F0_V is not None:
        F0_vals = np.abs(np.random.normal(F0_V,epsilon, pdf_smaple_N))
    else:
        F0_vals = np.abs(np.random.normal(F0_mean, F0_std, pdf_smaple_N))
        if F_zero_portion is not None:
            F0_vals_zero = np.zeros(int(F_zero_portion * pdf_smaple_N))
            F0_vals = np.concatenate((F0_vals_zero,np.abs(np.random.normal(F0_mean, F0_std, pdf_smaple_N - int(F_zero_portion * pdf_smaple_N)))))

    # eqs = ["dx/dt = v", "dv/dt = -k/m * x - c/m * v + F0/m * cos(omega*t)"]
    # coef_names = ["constant", "x", "v", "cos(omega*t)", "x**2", "v**2", "x*v"]

    # gt_coef = [{'constant': [0, 0], 'x': [0, 0], 'v': [1, 0], 'cos(omega*t)': [0, 0], 'x**2': [0, 0], 'v**2': [0, 0],
    #             'x*v': [0, 0]},
    #            {'constant': [0, 0], 'x': [real_params['-k/m_mean'], real_params['-k/m_std']],
    #             'v': [real_params['-c/m_mean'], real_params['-c/m_std']],
    #             'cos(omega*t)': [real_params['F0/m_mean'], real_params['F0/m_std']], 'x**2': [0, 0], 'v**2': [0, 0],
    #             'x*v': [0, 0]}]
    coef_list_1 = [np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #const
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #x
                   np.abs(np.random.normal(1,epsilon, pdf_smaple_N)), #v
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #cos
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #x**2
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #v**2
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N))] #x*v

    coef_list_2 = [np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #const
                   - k_vals/m_vals ,                                      #x
                   -c_vals/m_vals ,                                       #v
                   F0_vals/m_vals ,                                       #cos
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #x**2
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N)), #v**2
                   np.abs(np.random.normal(0,epsilon, pdf_smaple_N))] #x*v

    pdf_list = [coef_list_1,coef_list_2]
    pdf_arr = np.array(pdf_list)
    return pdf_arr