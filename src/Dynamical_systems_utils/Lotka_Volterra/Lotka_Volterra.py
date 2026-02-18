from scipy.integrate import solve_ivp
import numpy as np
import math
from src.pdf_utils import gen_param
import os
import pickle

class LotkaVolterra:
    """
    A class to simulate the Lotka-Volterra predator-prey system.
    """

    def __init__(self, alpha, beta, gamma, delta, x0=10.0, y0=10.0, gradient_targets = True):
        """
        Initializes the Lotka-Volterra parameters and initial conditions.

        Args:
            alpha (float): Growth rate of prey.
            beta (float): Predation rate.
            gamma (float): Death rate of predators.
            delta (float): Reproduction rate of predators per prey consumed.
            x0 (float): Initial population of prey (default: 10.0).
            y0 (float): Initial population of predators (default: 10.0).
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.x0 = x0
        self.y0 = y0
        self.gradient_targets = gradient_targets

    def _equations(self, t, state):
        """
        Defines the equations of motion for the Lotka-Volterra system.

        Args:
            t (float): Current time.
            state (numpy.ndarray): A NumPy array containing the current prey population (x) and predator population (y).

        Returns:
            numpy.ndarray: A NumPy array containing the derivatives dx/dt and dy/dt.
        """
        x, y = state
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return [dxdt, dydt]

    def simulate(self, t_span=(0, 20), dt=0.01, noise_level=0.01):
        """
        Simulates the Lotka-Volterra system over a given time span.

        Args:
            t_span (tuple): A tuple (t_start, t_end) defining the time interval of simulation (default: (0, 20)).
            dt (float): The time step for generating the output time points (default: 0.01).
            noise_std (float): The standard deviation of the Gaussian noise to add to the simulated data (default: 0.1).

        Returns:
            dict: A dictionary containing the time points ('t'), prey population ('x'), predator population ('y'),
                  and their derivatives ('dx_dt', 'dy_dt') with added noise.
        """
        sol = solve_ivp(self._equations, t_span, [self.x0, self.y0],
                        dense_output=True, method='RK45')
        t_eval = np.arange(t_span[0], t_span[1] , dt)
        results = sol.sol(t_eval)


        x = results[0]
        y = results[1]

        # # Calculate derivatives (without noise for the "true" Y)
        # true_dxdt = self.alpha * results[0] - self.beta * results[0] * results[1]
        # true_dydt = self.delta * results[0] * results[1] - self.gamma * results[1]

        # dx_dt = true_dxdt + np.random.normal(0, noise_std, len(t_eval))
        # dy_dt = true_dydt + np.random.normal(0, noise_std, len(t_eval))
        if self.gradient_targets:
            # Compute numerical gradients
            dx_dt = np.gradient(x, dt)
            dy_dt = np.gradient(y, dt)
        else:
            dx_dt = self.alpha * results[0] - self.beta * results[0] * results[1]
            dy_dt = self.delta * results[0] * results[1] - self.gamma * results[1]


        xnoise_scale = np.std(results[0]) * noise_level
        ynoise_scale = np.std(results[1]) * noise_level

        x = results[0] + np.random.normal(0, xnoise_scale, len(t_eval))
        y = results[1] + np.random.normal(0, ynoise_scale, len(t_eval))


        return {
            't': t_eval,
            'x': x,
            'y': y,
            'dx_dt': dx_dt,
            'dy_dt': dy_dt
        }
print("---------------------------- Mix Function -------------------------")
def mix_data(system_param_dict):
    N_param_set = system_param_dict['N_param_set']
    alpha_V = system_param_dict['alpha_info']['alpha_V'] if 'alpha_V' in system_param_dict['alpha_info'] else None
    alpha_mean = system_param_dict['alpha_info']['alpha_mean'] if 'alpha_mean' in system_param_dict['alpha_info'] else None
    alpha_std = system_param_dict['alpha_info']['alpha_std'] if 'alpha_std' in system_param_dict['alpha_info'] else None
    alpha_N = system_param_dict['alpha_info']['alpha_N'] if 'alpha_N' in system_param_dict['alpha_info'] else None

    beta_V = system_param_dict['beta_info']['beta_V'] if 'beta_V' in system_param_dict['beta_info'] else None
    beta_mean = system_param_dict['beta_info']['beta_mean'] if 'beta_mean' in system_param_dict['beta_info'] else None
    beta_std = system_param_dict['beta_info']['beta_std'] if 'beta_std' in system_param_dict['beta_info'] else None
    beta_N = system_param_dict['beta_info']['beta_N'] if 'beta_N' in system_param_dict['beta_info'] else None

    gamma_V = system_param_dict['gamma_info']['gamma_V'] if 'gamma_V' in system_param_dict['gamma_info'] else None
    gamma_mean = system_param_dict['gamma_info']['gamma_mean'] if 'gamma_mean' in system_param_dict['gamma_info'] else None
    gamma_std = system_param_dict['gamma_info']['gamma_std'] if 'gamma_std' in system_param_dict['gamma_info'] else None
    gamma_N = system_param_dict['gamma_info']['gamma_N'] if 'gamma_N' in system_param_dict['gamma_info'] else None

    delta_V = system_param_dict['delta_info']['delta_V'] if 'delta_V' in system_param_dict['delta_info'] else None
    delta_mean = system_param_dict['delta_info']['delta_mean'] if 'delta_mean' in system_param_dict['delta_info'] else None
    delta_std = system_param_dict['delta_info']['delta_std'] if 'delta_std' in system_param_dict['delta_info'] else None
    delta_N = system_param_dict['delta_info']['delta_N'] if 'delta_N' in system_param_dict['delta_info'] else None

    t_start = system_param_dict['t_info']['t_start'] if 't_start' in system_param_dict['t_info'] else 0
    t_end = system_param_dict['t_info']['t_end'] if 't_end' in system_param_dict['t_info'] else 20
    dt = system_param_dict['t_info']['dt'] if 'dt' in system_param_dict['t_info'] else 0.01
    noise_level = system_param_dict['noise_info']["noise_level"] if "noise_level" in list(
        system_param_dict["noise_info"].keys()) else 0.01
    x0_V = system_param_dict['x0_info']['x0_V'] if 'x0_V' in system_param_dict['x0_info'] else 10
    y0_V = system_param_dict['y0_info']['y0_V'] if 'y0_V' in system_param_dict['y0_info'] else 10


    X_all = []
    Y_all = []

    alpha_list = []
    beta_list = []
    gamma_list = []
    delta_list = []

    alpha_gen = gen_param(N_param_set,alpha_V, alpha_mean, alpha_std)
    beta_gen = gen_param(N_param_set, beta_V, beta_mean, beta_std)
    gamma_gen = gen_param(N_param_set, gamma_V, gamma_mean, gamma_std)
    delta_gen = gen_param(N_param_set, delta_V, delta_mean, delta_std)

    for param_set in range(N_param_set):
        alpha = math.fabs(alpha_gen.gen())
        beta = math.fabs(beta_gen.gen())
        gamma = math.fabs(gamma_gen.gen())
        delta = math.fabs(delta_gen.gen())

        alpha_list.append(alpha)
        beta_list.append(-beta)
        gamma_list.append(-gamma)
        delta_list.append(delta)


        oscillator = LotkaVolterra(alpha=alpha, beta=beta, gamma=gamma, delta=delta, x0=x0_V if x0_V is not None else 10.0, y0=y0_V if y0_V is not None else 10.0)
        results = oscillator.simulate(t_span=(t_start, t_end), dt=dt, noise_level=noise_level)

        t = results['t']
        x = results['x']
        y = results['y']
        dx_dt = results['dx_dt']
        dy_dt = results['dy_dt']

        # X contains [x, y, x*y] for SINDy-like regression for Lotka-Volterra
        X = np.array([np.ones_like(x), x, y, x * y, x**2, y**2])
        Y = np.array([dx_dt, dy_dt])

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    real_params = {
        'alpha_mean': np.mean(np.array(alpha_list)),
        'alpha_std': np.std(np.array(alpha_list)),
        'beta_mean': np.mean(np.array(beta_list)),
        'beta_std': np.std(np.array(beta_list)),
        'gamma_mean': np.mean(np.array(gamma_list)),
        'gamma_std': np.std(np.array(gamma_list)),
        'delta_mean': np.mean(np.array(delta_list)),
        'delta_std': np.std(np.array(delta_list)),

        'alpha_array': np.array(alpha_list),
        'beta_array': np.array(beta_list),
        'gamma_array': np.array(gamma_list),
        'delta_array': np.array(delta_list),
    }

    return X_all, Y_all, real_params

def gt_utils(real_params):
    eqs = ["dx/dt = alpha * x - beta * x * y", "dy/dt = delta * x * y - gamma * y"]
    coef_names = ["const","x", "y", "x*y","x**2","y**2"]

    gt_coef_dx = {'const':[0,0],
         'x': [real_params['alpha_mean'], real_params['alpha_std']],
         'y': [0, 0],
         'x*y': [real_params['beta_mean'], real_params['beta_std']],
         "x**2":[0,0],"y**2":[0,0]
                  }

    # Ground truth coefficients for the dw/dt equation
    # [constant (a*b0), v (a*b1), w (-a), v^3, v^2, w^2, v*w]
    gt_coef_dy = {'const':[0,0],
                  'x': [0, 0],
                  'y': [real_params['gamma_mean'], real_params['gamma_std']],
                  'x*y': [real_params['delta_mean'], real_params['delta_std']],
                  "x**2":[0,0],
                  "y**2":[0,0]
                  }

    # Combine into a list for dv/dt and dw/dt
    gt_coef = [gt_coef_dx, gt_coef_dy]
    eq_list = []
    for eq in gt_coef:
        coef_list = []
        for k, v in eq.items():
            coef_list += [v]
        eq_list += [coef_list]
    gt_info_arr = np.array(eq_list)
    return {"eqs": eqs, "coef_names": coef_names, "gt_coef": gt_coef, "gt_info_arr": gt_info_arr}


def realparame2gtarray(real_params:dict):
    dxdt_const_arr = np.zeros_like(real_params['alpha_array'])
    dxdt_x_arr = real_params['alpha_array']
    dxdt_y_arr = np.zeros_like(real_params['alpha_array'])
    dxdt_xy_arr = real_params['beta_array']
    dxdt_x2_arr = np.zeros_like(real_params['alpha_array'])
    dxdt_y2_arr = np.zeros_like(real_params['alpha_array'])


    dvdt_const_arr = np.zeros_like(real_params['alpha_array'])
    dvdt_x_arr = np.zeros_like(real_params['alpha_array'])
    dvdt_y_arr = real_params['gamma_array']
    dvdt_xy_arr = real_params['delta_array']
    dvdt_x2_arr = np.zeros_like(real_params['alpha_array'])
    dvdt_y2_arr = np.zeros_like(real_params['alpha_array'])

    # Correcting concatenation syntax
    gt_arr1 = np.array([dxdt_const_arr,
                        dxdt_x_arr,
                        dxdt_y_arr,
                        dxdt_xy_arr,
                        dxdt_x2_arr,
                        dxdt_y2_arr
                        ])
    gt_arr2 = np.array([dvdt_const_arr,
                        dvdt_x_arr,
                        dvdt_y_arr,
                        dvdt_xy_arr,
                        dvdt_x2_arr,
                        dvdt_y2_arr
                        ])

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

    """
    # eqs = ["dx/dt = alpha * x - beta * x * y", "dy/dt = delta * x * y - gamma * y"]
    # coef_names = ["const","x", "y", "x*y","x**2","y**2"]

    # gt_coef = [
    #     {'const':[0,0],'x': [real_params['alpha_mean'], real_params['alpha_std']], 'y': [0, 0], 'x*y': [real_params['beta_mean'], real_params['beta_std']], "x**2":[0,0],"y**2":[0,0]},
    #     {'const':[0,0],'x': [0, 0], 'y': [real_params['gamma_mean'], real_params['gamma_std']], 'x*y': [real_params['delta_mean'], real_params['delta_std']], "x**2":[0,0],"y**2":[0,0]}
    # ]
    with open(os.path.join(save_path, "system_param_dict.pkl"), "rb") as f:
        system_param_dict = pickle.load(f)

    alpha_V = system_param_dict['alpha_info']['alpha_V'] if 'alpha_V' in system_param_dict['alpha_info'] else None
    alpha_mean = system_param_dict['alpha_info']['alpha_mean'] if 'alpha_mean' in system_param_dict[
        'alpha_info'] else None
    alpha_std = system_param_dict['alpha_info']['alpha_std'] if 'alpha_std' in system_param_dict['alpha_info'] else None
    alpha_N = system_param_dict['alpha_info']['alpha_N'] if 'alpha_N' in system_param_dict['alpha_info'] else None

    beta_V = system_param_dict['beta_info']['beta_V'] if 'beta_V' in system_param_dict['beta_info'] else None
    beta_mean = system_param_dict['beta_info']['beta_mean'] if 'beta_mean' in system_param_dict['beta_info'] else None
    beta_std = system_param_dict['beta_info']['beta_std'] if 'beta_std' in system_param_dict['beta_info'] else None
    beta_N = system_param_dict['beta_info']['beta_N'] if 'beta_N' in system_param_dict['beta_info'] else None

    gamma_V = system_param_dict['gamma_info']['gamma_V'] if 'gamma_V' in system_param_dict['gamma_info'] else None
    gamma_mean = system_param_dict['gamma_info']['gamma_mean'] if 'gamma_mean' in system_param_dict[
        'gamma_info'] else None
    gamma_std = system_param_dict['gamma_info']['gamma_std'] if 'gamma_std' in system_param_dict['gamma_info'] else None
    gamma_N = system_param_dict['gamma_info']['gamma_N'] if 'gamma_N' in system_param_dict['gamma_info'] else None

    delta_V = system_param_dict['delta_info']['delta_V'] if 'delta_V' in system_param_dict['delta_info'] else None
    delta_mean = system_param_dict['delta_info']['delta_mean'] if 'delta_mean' in system_param_dict[
        'delta_info'] else None
    delta_std = system_param_dict['delta_info']['delta_std'] if 'delta_std' in system_param_dict['delta_info'] else None


    alpha_list = []
    beta_list = []
    gamma_list = []
    delta_list = []

    alpha_gen = gen_param(pdf_smaple_N, alpha_V, alpha_mean, alpha_std)
    beta_gen = gen_param(pdf_smaple_N, beta_V, beta_mean, beta_std)
    gamma_gen = gen_param(pdf_smaple_N, gamma_V, gamma_mean, gamma_std)
    delta_gen = gen_param(pdf_smaple_N, delta_V, delta_mean, delta_std)

    for param_set in range(pdf_smaple_N):
        alpha = math.fabs(alpha_gen.gen())
        beta = math.fabs(beta_gen.gen())
        gamma = math.fabs(gamma_gen.gen())
        delta = math.fabs(delta_gen.gen())

        alpha_list.append(alpha)
        beta_list.append(-beta)
        gamma_list.append(-gamma)
        delta_list.append(delta)
    # eqs = ["dx/dt = alpha * x - beta * x * y", "dy/dt = delta * x * y - gamma * y"]
    # coef_names = ["const","x", "y", "x*y","x**2","y**2"]

    coef_0 = [np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
              np.array(alpha_list),
              np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
              np.array(beta_list),
              np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
              np.abs(np.random.normal(0, epsilon, pdf_smaple_N))]

    coef_1 = [np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
              np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
              np.array(gamma_list),
              np.array(delta_list),
              np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),
              np.abs(np.random.normal(0, epsilon, pdf_smaple_N))]

    pdf_arr = np.array([coef_0,coef_1])
    return pdf_arr
