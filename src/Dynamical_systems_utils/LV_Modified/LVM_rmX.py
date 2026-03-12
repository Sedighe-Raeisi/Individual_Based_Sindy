import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.pdf_utils import gen_param
import math
import os
import pickle

########################## In this code we remove the x from library of candidate terms to prevent overlap with term x(1-x) ############################

# links for parameter selection:
# perplexity : https://www.perplexity.ai/search/what-parameters-are-to-be-used-yEfZVv4DSyOkJp1WRjf8Sw
# Gemini : https://gemini.google.com/share/25853a174273

# --- Provided LV_M class (do not change) ---
class LV_M:
    def __init__(self, alpha=1.0, beta=1.6, h=0.4, epsilon=0.1, m=0.4, H=0.075, gradient_targets=True):
        self.alpha, self.beta, self.h = alpha, beta, h
        self.epsilon, self.m, self.H = epsilon, m, H
        self.gradient_targets = gradient_targets

    def _equations(self, t, state):
        x, y = state
        # Prey Equation with Logistic Growth (g2(x)) and Type II Response (f2(x))
        # dxdt = alpha * (1 - x) * x - (beta * x * y) / (1 + h * x)
        dxdt = self.alpha * (1 - x) * x - (self.beta * x * y) / (1 + self.h * x)

        # Predator Equation with Type II Response (f2(x)) and Quadratic Mortality (h3(Y))
        # dydt = (epsilon * beta * x * y) / (1 + h * x) - m * y - H * y ** 2
        dydt = (self.epsilon * self.beta * x * y) / (1 + self.h * x) - self.m * y - self.H * y ** 2
        return [dxdt, dydt]

    def simulate(self, x0, y0, t_span=(0, 100), N_t=1000, noise_level=0.01):
        sol = solve_ivp(self._equations, t_span, [x0, y0], dense_output=True, method='RK45', rtol=1e-9)
        dt = (t_span[1] - t_span[0]) / N_t
        t_eval = np.arange(t_span[0], t_span[1], dt)
        results = sol.sol(t_eval)

        x = results[0]
        y = results[1]

        # Targets based on internal, clean solution (per class definition)
        if self.gradient_targets:
            dx_dt = np.gradient(x, dt)
            dy_dt = np.gradient(y, dt)
        else:
            dx_dt = self.alpha * (1 - x) * x - (self.beta * x * y) / (1 + self.h * x)
            dy_dt = (self.epsilon * self.beta * x * y) / (1 + self.h * x) - self.m * y - self.H * y ** 2

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


# --- End of provided class ---


def mix_data_LV_M(system_param_dict):
    """
    Revised Mix function to include all terms from the blackboard in the library.
    Citations: Photo (coef_names, library terms), LV_M class (simulation)
    """
    N_param_set = system_param_dict['N_param_set']

    # --- Parameter setup ---
    # We retrieve delta parameters if present, though they will be unused
    # as LV_M doesn't contain terms h4(Y) or h5(Y).
    def get_param_info(name):
        return (system_param_dict.get(f'{name}_info', {}).get(f'{name}_V'),
                system_param_dict.get(f'{name}_info', {}).get(f'{name}_mean'),
                system_param_dict.get(f'{name}_info', {}).get(f'{name}_std'))

    param_keys = ['alpha', 'beta', 'h', 'epsilon', 'm', 'H', 'delta1', 'delta2', 'delta3', 'delta4']
    gens = {p: gen_param(N_param_set, *get_param_info(p)) for p in param_keys}

    # Store generated lists
    p_lists = {p: [] for p in param_keys}

    # --- Time and Initial Conditions ---
    noise_level = system_param_dict.get('noise_info',{}).get("noise_level",0.01)
    t_info = system_param_dict.get('t_info', {})
    t_start = t_info.get('t_start', 0)
    t_end = t_info.get('t_end', 20)
    N_t = t_info.get('N_t', 1000)
    dt = (t_end - t_start) / N_t

    x0 = system_param_dict.get('x0_info', {}).get('x0_V', 10)
    y0 = system_param_dict.get('y0_info', {}).get('y0_V', 10)

    # --- Storage ---
    X_all = []
    Y_all = []

    # Simulation loop
    for _ in range(N_param_set):
        # Generate positive parameters for this set

        current_p = {p: abs(gens[p].gen()) for p in param_keys}
        for p in param_keys:
            p_lists[p].append(current_p[p])

        # 1. Simulate using the *exact* LV_M model definition
        model = LV_M(alpha=current_p['alpha'], beta=current_p['beta'], h=current_p['h'],
                     epsilon=current_p['epsilon'], m=current_p['m'], H=current_p['H'])

        # By setting noise_level to 0.0, we get the exact 'x' and 'y' required
        # for regression against analytical derivatives in the library.
        sol = model.simulate(x0=x0, y0=y0, t_span=(t_start, t_end), N_t=N_t, noise_level=noise_level)

        # We need the exact analytical derivative for Y
        x, y = sol['x'], sol['y']
        dxdt = sol['dx_dt']
        dydt = sol['dy_dt']
        # Compute exact analytical derivatives for regression target
        # dx_dt = alpha * x - alpha * x**2 - (beta * x * y) / (1 + h * x)
        dxdt_true = current_p['alpha'] * x * (1 - x) - (current_p['beta'] * x * y) / (
                    1 + current_p['h'] * x)

        # dydt = (epsilon * beta * x * y) / (1 + h * x) - m * y - H * y ** 2
        eb_product = current_p['epsilon'] * current_p['beta']
        dydt_true = (eb_product * x * y) / (1 + current_p['h'] * x) - current_p['m'] * y - \
                    current_p['H'] * y ** 2

        # --- Expanded Library (X array) from the photo ---
        # Coef Names: ["H (const)", "g1(x)=alpha*x", "g2(x)=alpha*x(1-x)", "g3(x)", "Y (h2)", "Y^2 (h3)", "x*Y", "(x*Y)/(1+h*x)"]
        # We must align X terms *exactly* with the names/formulas in gt_utils.

        # Common factors for rational functions
        den_f2 = 1 + current_p['h'] * x

        X = np.array([
            np.ones_like(x),  # 0. H (const)
            # x,  # 1. g1(x) = alpha*x (only the x part)
            x * (1 - x),  # 2. g2(x) = alpha*x(1-x) (the x*(1-x) part)
            x ** 2 * (1 - x),  # 3. Placeholder for g3(x)
            y,  # 4. Y (h2)
            y ** 2,  # 5. Y^2 (h3)
            x * y,  # 6. x*Y (base linear interaction f1)
            (x * y) / den_f2,  # 7. f2(x)*Y = (beta*x*y)/(1+h*x)
            (x ** 2 * y) / (1 + current_p['h'] * x ** 2),  # 8. f3(x)*Y = (beta*x^2*y)/(1+h*x^2)
            y / (1 + current_p['delta2'] * y),  # 9. h4(Y)
            y ** 2 / (1 + current_p['delta4'] * y ** 2),  # 10. h5(Y)
        ])

        Y_target = np.array([dxdt, dydt])

        X_all.append(X)
        Y_all.append(Y_target)

    # Convert lists to numpy arrays
    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    # Store collected parameter arrays for real_params
    p_arrays = {f'{p}_array': np.array(p_lists[p]) for p in param_keys}

    return X_all, Y_all, p_arrays


def gt_utils(real_params):
    """
    Revised ground truth utility function.
    coef_names now include ALL terms from the photo.
    Values are mapped based on the specific LV_M model implementation.
    Citations: Photo (equations, names), LV_M class (logic)
    """

    # Define the *general* form of equations as drawn
    eqs = [
        "dx/dt = g(x)X - f(x)Y",
        "dy/dt = epsilon*f(x)Y - mY - h(Y)"
    ]

    # 1. List ALL potential term names identified from the photo
    coef_names = [
        "H (const)",  # 0. Constant term h1(Y)
        # "g1(x)=alpha*x",  # 1. Basic prey growth
        "g2(x)=alpha*x(1-x)",  # 2. Logistic prey growth
        "g3(x)",  # 3. Third prey polynomial placeholder
        "Y (h2)",  # 4. Linear mortality
        "Y^2 (h3)",  # 5. Quadratic mortality
        "x*Y",  # 6. Linear interaction f1
        "(x*Y)/(1+h*x)",  # 7. Type II interaction f2
        "(x^2*Y)/(1+h*x^2)",  # 8. Type III interaction f3
        "Y/(1+delta2*Y)",  # 9. Ratio mortality h4
        "Y^2/(1+delta4*Y^2)",  # 10. Ratio mortality h5
    ]

    # Helper function to get mean/std from an array
    def get_stats(arr):
        return [np.mean(arr), np.std(arr)]

    # 2. Map ground truth values based on the specific *LV_M* implementation

    # dx/dt = alpha*x*(1-x) - beta*xy/(1+hx)
    gt_coef_dx = {name: [0, 0] for name in coef_names}
    gt_coef_dx["g2(x)=alpha*x(1-x)"] = get_stats(real_params['alpha_array'])
    gt_coef_dx["(x*Y)/(1+h*x)"] = [-np.mean(real_params['beta_array']),
                                   np.std(real_params['beta_array'])]  # NOTE: negative sign for consumption

    # dy/dt = epsilon*beta*xy/(1+hx) - m*y - H*y^2
    gt_coef_dy = {name: [0, 0] for name in coef_names}
    # Combined parameter epsilon*beta
    eb_mean = np.mean(real_params['epsilon_array'] * real_params['beta_array'])
    eb_std = np.std(real_params['epsilon_array'] * real_params['beta_array'])
    gt_coef_dy["(x*Y)/(1+h*x)"] = [eb_mean, eb_std]

    # Mortonality terms in h(Y) = mY + h2 + h3... are *subtracted* in dy/dt
    gt_coef_dy["Y (h2)"] = [-np.mean(real_params['m_array']), np.std(real_params['m_array'])]
    gt_coef_dy["Y^2 (h3)"] = [-np.mean(real_params['H_array']), np.std(real_params['H_array'])]

    # Combine into required list format
    gt_coef = [gt_coef_dx, gt_coef_dy]

    eq_list = []
    for eq_dict in gt_coef:
        coef_list = []
        for name in coef_names:  # Must iterate in fixed order
            coef_list.append(eq_dict[name])
        eq_list.append(coef_list)

    gt_info_arr = np.array(eq_list)
    return {"eqs": eqs, "coef_names": coef_names, "gt_coef": gt_coef, "gt_info_arr": gt_info_arr}


def realparame2gtarray(real_params: dict):
    """
    Revised mapping function to return the true coefficient arrays across parameter sets.
    Citations: Photo (names/library structure), LV_M class (logic)
    """

    # Get any one parameter array to determine the size
    alpha_arr = real_params['alpha_array']
    N_sets = len(alpha_arr)
    zeros = np.zeros(N_sets)
    ones = np.ones(N_sets)

    # 1. Map true arrays for dx/dt = alpha*x*(1-x) - beta*xy/(1+hx)
    # Names match gt_utils: ["H (const)", "g1", "g2", "g3", "h2", "h3", "f1", "f2", "f3", "h4", "h5"]
    dxdt_true_arrs = [
        zeros,  # 0. H (const)
        # zeros,  # 1. g1(x)=alpha*x
        real_params['alpha_array'],  # 2. g2(x)=alpha*x(1-x)
        zeros,  # 3. g3(x)
        zeros,  # 4. Y (h2)
        zeros,  # 5. Y^2 (h3)
        zeros,  # 6. x*Y
        -real_params['beta_array'],  # 7. (x*Y)/(1+h*x) [negative]
        zeros,  # 8. f3
        zeros,  # 9. h4
        zeros,  # 10. h5
    ]

    # 2. Map true arrays for dy/dt = epsilon*beta*xy/(1+hx) - m*y - H*y^2
    # Combined parameter product
    eb_arr = real_params['epsilon_array'] * real_params['beta_array']

    dvdt_true_arrs = [
        zeros,  # 0. H (const)
        # zeros,  # 1. g1
        zeros,  # 2. g2
        zeros,  # 3. g3
        -real_params['m_array'],  # 4. Y (h2)
        -real_params['H_array'],  # 5. Y^2 (h3)
        zeros,  # 6. x*Y
        eb_arr,  # 7. (x*Y)/(1+h*x)
        zeros,  # 8. f3
        zeros,  # 9. h4
        zeros,  # 10. h5
    ]

    gt_arr = np.array([dxdt_true_arrs, dvdt_true_arrs])
    return gt_arr




def generate_pdf(save_path, pdf_smaple_N=10000, epsilon=0.01):
    """
    Generates a PDF array for the LV_M system.
    Output shape: (2, 11, pdf_smaple_N)
    representing (N_equations, N_coefficients, N_samples).
    """

    with open(os.path.join(save_path, "system_param_dict.pkl"), "rb") as f:
        system_param_dict = pickle.load(f)

    # List of parameters required by the LV_M model
    param_keys = ['alpha', 'beta', 'h', 'epsilon', 'm', 'H', 'delta1', 'delta2', 'delta3', 'delta4']

    # Initialize generators for all parameters
    gens = {}
    for p in param_keys:
        p_info = system_param_dict.get(f'{p}_info', {})
        # Assuming gen_param is defined globally as in your snippet
        gens[p] = gen_param(
            pdf_smaple_N,
            p_info.get(f'{p}_V'),
            p_info.get(f'{p}_mean'),
            p_info.get(f'{p}_std')
        )

    # Generate samples and ensure they are positive (using math.fabs)
    samples = {p: [math.fabs(gens[p].gen()) for _ in range(pdf_smaple_N)] for p in param_keys}

    # Convert to numpy arrays for vectorization
    p = {k: np.array(v) for k, v in samples.items()}

    # --- Coefficient Mapping for Equation 1 (dx/dt) ---
    # Model: dx/dt = alpha * x(1-x) - (beta * x * y) / (1 + h * x)
    # Library Index: 0:const, 1:x, 2:x(1-x), 3:x^2(1-x), 4:y, 5:y^2, 6:xy, 7:xy/(1+hx), ...

    noise_vec = lambda: np.random.normal(0, epsilon, pdf_smaple_N)

    coef_0 = [
        noise_vec(),  # 0: H (const)
        # noise_vec(),  # 1: g1(x) = x
        p['alpha'],  # 2: g2(x) = x(1-x)  <-- Target Term
        noise_vec(),  # 3: g3(x)
        noise_vec(),  # 4: Y
        noise_vec(),  # 5: Y^2
        noise_vec(),  # 6: x*Y
        -p['beta'],  # 7: (x*Y)/(1+h*x)   <-- Target Term (Negative)
        noise_vec(),  # 8: f3(x)*Y
        noise_vec(),  # 9: h4(Y)
        noise_vec()  # 10: h5(Y)
    ]

    # --- Coefficient Mapping for Equation 2 (dy/dt) ---
    # Model: dy/dt = (epsilon * beta * x * y) / (1 + h * x) - m * y - H * y^2

    coef_1 = [
        noise_vec(),  # 0: const
        # noise_vec(),  # 1: x
        noise_vec(),  # 2: x(1-x)
        noise_vec(),  # 3: x^2(1-x)
        -p['m'],  # 4: Y               <-- Target Term (Negative)
        -p['H'],  # 5: Y^2             <-- Target Term (Negative)
        noise_vec(),  # 6: x*Y
        p['epsilon'] * p['beta'],  # 7: (x*Y)/(1+h*x)  <-- Target Term
        noise_vec(),  # 8: f3(x)*Y
        noise_vec(),  # 9: h4(Y)
        noise_vec()  # 10: h5(Y)
    ]

    # Combine into (2, 11, pdf_sample_N)
    return np.array([coef_0, coef_1])


# class LV_M:
#     def __init__(self, alpha=1.0, beta=2.0, h=0.5, epsilon=0.5, m=0.3, H=0.05,gradient_targets=True):
#         # DO NOT CHANGE THESE FUNCTIONS - Per user request
#         self.alpha, self.beta, self.h = alpha, beta, h
#         self.epsilon, self.m, self.H = epsilon, m, H
#         self.gradient_targets = gradient_targets
#     def _equations(self, t, state):
#         x, y = state
#         # Prey Equation with Logistic Growth and Type II Response
#         dxdt = self.alpha * (1 - x) * x - (self.beta * x * y) / (1 + self.h * x)
#         # Predator Equation with Type II Response and Quadratic Mortality
#         dydt = (self.epsilon * self.beta * x * y) / (1 + self.h * x) - self.m * y - self.H * y ** 2
#         return [dxdt, dydt]
#
#     def simulate(self, x0, y0, t_span=(0, 100),N_t = 1000, noise_level=0.01):
#         sol = solve_ivp(self._equations, t_span, [x0, y0], dense_output=True, method='RK45', rtol=1e-9)
#         # t_eval = np.linspace(t_span[0], t_span[1], N_t)
#         dt = (t_span[1]-t_span[0])/N_t
#         t_eval = np.arange(t_span[0], t_span[1], dt)
#         results = sol.sol(t_eval)
#
#         x = results[0]
#         y = results[1]
#
#         if self.gradient_targets:
#             # Compute numerical gradients
#             dx_dt = np.gradient(x, dt)
#             dy_dt = np.gradient(y, dt)
#         else:
#             dxdt = self.alpha * (1 - x) * x - (self.beta * x * y) / (1 + self.h * x)
#             # Predator Equation with Type II Response and Quadratic Mortality
#             dydt = (self.epsilon * self.beta * x * y) / (1 + self.h * x) - self.m * y - self.H * y ** 2
#         xnoise_scale = np.std(results[0]) * noise_level
#         ynoise_scale = np.std(results[1]) * noise_level
#
#         x = results[0] + np.random.normal(0, xnoise_scale, len(t_eval))
#         y = results[1] + np.random.normal(0, ynoise_scale, len(t_eval))
#
#         return {
#             't': t_eval,
#             'x': x,
#             'y': y,
#             'dx_dt': dx_dt,
#             'dy_dt': dy_dt
#         }
#
#
#
# #########################################################################################
# ################################## Mix function #########################################
#
#
# def mix_data_LV_M(system_param_dict):
#
#     N_param_set = system_param_dict['N_param_set']
#
#     # ----- Alpha -----
#     alpha_V   = system_param_dict['alpha_info']['alpha_V']   if 'alpha_V'   in system_param_dict['alpha_info'] else None
#     alpha_mean = system_param_dict['alpha_info']['alpha_mean'] if 'alpha_mean' in system_param_dict['alpha_info'] else None
#     alpha_std  = system_param_dict['alpha_info']['alpha_std']  if 'alpha_std'  in system_param_dict['alpha_info'] else None
#
#     # ----- Beta -----
#     beta_V   = system_param_dict['beta_info']['beta_V']   if 'beta_V'   in system_param_dict['beta_info'] else None
#     beta_mean = system_param_dict['beta_info']['beta_mean'] if 'beta_mean' in system_param_dict['beta_info'] else None
#     beta_std  = system_param_dict['beta_info']['beta_std']  if 'beta_std'  in system_param_dict['beta_info'] else None
#
#     # ----- h -----
#     h_V   = system_param_dict['h_info']['h_V']   if 'h_V'   in system_param_dict['h_info'] else None
#     h_mean = system_param_dict['h_info']['h_mean'] if 'h_mean' in system_param_dict['h_info'] else None
#     h_std  = system_param_dict['h_info']['h_std']  if 'h_std'  in system_param_dict['h_info'] else None
#
#     # ----- epsilon -----
#     epsilon_V   = system_param_dict['epsilon_info']['epsilon_V']   if 'epsilon_V'   in system_param_dict['epsilon_info'] else None
#     epsilon_mean = system_param_dict['epsilon_info']['epsilon_mean'] if 'epsilon_mean' in system_param_dict['epsilon_info'] else None
#     epsilon_std  = system_param_dict['epsilon_info']['epsilon_std']  if 'epsilon_std'  in system_param_dict['epsilon_info'] else None
#
#     # ----- m -----
#     m_V   = system_param_dict['m_info']['m_V']   if 'm_V'   in system_param_dict['m_info'] else None
#     m_mean = system_param_dict['m_info']['m_mean'] if 'm_mean' in system_param_dict['m_info'] else None
#     m_std  = system_param_dict['m_info']['m_std']  if 'm_std'  in system_param_dict['m_info'] else None
#
#     # ----- H -----
#     H_V   = system_param_dict['H_info']['H_V']   if 'H_V'   in system_param_dict['H_info'] else None
#     H_mean = system_param_dict['H_info']['H_mean'] if 'H_mean' in system_param_dict['H_info'] else None
#     H_std  = system_param_dict['H_info']['H_std']  if 'H_std'  in system_param_dict['H_info'] else None
#
#     # ----- Time -----
#     t_start = system_param_dict['t_info']['t_start'] if 't_start' in system_param_dict['t_info'] else 0
#     t_end   = system_param_dict['t_info']['t_end']   if 't_end'   in system_param_dict['t_info'] else 100
#     N_t     = system_param_dict['t_info']['N_t']     if 'N_t'     in system_param_dict['t_info'] else 1000
#
#     # ----- Initial Conditions -----
#     x0 = system_param_dict['x0_info']['x0_V'] if 'x0_V' in system_param_dict['x0_info'] else 0.5
#     y0 = system_param_dict['y0_info']['y0_V'] if 'y0_V' in system_param_dict['y0_info'] else 0.5
#
#     # Storage
#     X_all = []
#     Y_all = []
#
#     alpha_list, beta_list, h_list = [], [], []
#     epsilon_list, m_list, H_list = [], [], []
#
#     # Parameter generators
#     alpha_gen   = gen_param(N_param_set, alpha_V, alpha_mean, alpha_std)
#     beta_gen    = gen_param(N_param_set, beta_V, beta_mean, beta_std)
#     h_gen       = gen_param(N_param_set, h_V, h_mean, h_std)
#     epsilon_gen = gen_param(N_param_set, epsilon_V, epsilon_mean, epsilon_std)
#     m_gen       = gen_param(N_param_set, m_V, m_mean, m_std)
#     H_gen       = gen_param(N_param_set, H_V, H_mean, H_std)
#
#     for _ in range(N_param_set):
#
#         alpha   = abs(alpha_gen.gen())
#         beta    = abs(beta_gen.gen())
#         h       = abs(h_gen.gen())
#         epsilon = abs(epsilon_gen.gen())
#         m       = abs(m_gen.gen())
#         H       = abs(H_gen.gen())
#
#         alpha_list.append(alpha)
#         beta_list.append(beta)
#         h_list.append(h)
#         epsilon_list.append(epsilon)
#         m_list.append(m)
#         H_list.append(H)
#
#         # Simulate system
#         model = LV_M(alpha=alpha, beta=beta, h=h,
#                      epsilon=epsilon, m=m, H=H)
#
#         sol = model.simulate(x0=x0, y0=y0,
#                              t_span=(t_start, t_end),
#                              N_t=N_t)
#
#         x = sol[0]
#         y = sol[1]
#
#         # Compute derivatives analytically (important for regression)
#         dx_dt = alpha * (1 - x) * x - (beta * x * y) / (1 + h * x)
#         dy_dt = (epsilon * beta * x * y) / (1 + h * x) - m * y - H * y**2
#
#         # Library for SINDy-style regression ???????????????????????????????????????????????????????????????????????????
#         X = np.array([                             # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#             np.ones_like(x),
#             x,
#             y,
#             x*y,
#             x**2,
#             y**2,
#             x/(1 + h*x),
#             (x*y)/(1 + h*x)
#         ])
#
#         Y = np.array([dx_dt, dy_dt])
#
#         X_all.append(X)
#         Y_all.append(Y)
#
#     X_all = np.array(X_all)
#     Y_all = np.array(Y_all)
#
#     real_params = {
#         'alpha_mean': np.mean(alpha_list),
#         'alpha_std':  np.std(alpha_list),
#         'beta_mean':  np.mean(beta_list),
#         'beta_std':   np.std(beta_list),
#         'h_mean':     np.mean(h_list),
#         'h_std':      np.std(h_list),
#         'epsilon_mean': np.mean(epsilon_list),
#         'epsilon_std':  np.std(epsilon_list),
#         'm_mean':     np.mean(m_list),
#         'm_std':      np.std(m_list),
#         'H_mean':     np.mean(H_list),
#         'H_std':      np.std(H_list),
#
#         'alpha_array':   np.array(alpha_list),
#         'beta_array':    np.array(beta_list),
#         'h_array':       np.array(h_list),
#         'epsilon_array': np.array(epsilon_list),
#         'm_array':       np.array(m_list),
#         'H_array':       np.array(H_list),
#     }
#
#     return X_all, Y_all, real_params






#
# ########################## Plot ######################################################
#
# def plot_journal_style(model):
#     fig, ax = plt.subplots(figsize=(7, 9))
#
#     # 1. Nullclines (Dashed brown lines)
#     x_vals = np.linspace(0.001, 1.5, 500)
#     # Prey Nullcline: y = [alpha * (1-x) * (1+hx)] / beta
#     prey_null = (model.alpha * (1 - x_vals) * (1 + model.h * x_vals)) / model.beta
#     # Predator Nullcline: y = ( (epsilon*beta*x)/(1+hx) - m ) / H
#     pred_null = (model.epsilon * model.beta * x_vals / (1 + model.h * x_vals) - model.m) / model.H
#     pred_null = np.maximum(0, pred_null)  # Physical constraint
#
#     ax.plot(x_vals, prey_null, color='#8c6d31', linestyle='--', linewidth=2, alpha=0.8)
#     ax.plot(x_vals, pred_null, color='#8c6d31', linestyle='--', linewidth=2, alpha=0.8)
#
#     # 2. Trajectories (Matching the colors in your image)
#     # Blue Trajectories (Outer)
#     blue_starts = [[1.5, 0.5], [0.15, 0.5]]
#     for s in blue_starts:
#         path = model.simulate(s[0], s[1], t_span=(0, 50))
#         ax.plot(path["x"], path["y"], color='blue', linewidth=1.5)
#         # Directional arrow
#         ax.annotate('', xy=(path["x"][100], path["y"][100]), xytext=(path[0][90], path[1][90]),
#                     arrowprops=dict(arrowstyle='->', color='blue', lw=2))
#
#     # Red Trajectory (Middle)
#     red_path = model.simulate(0.3, 1.3, t_span=(0, 40))
#     ax.plot(red_path["x"], red_path["y"], color='red', linewidth=1.5)
#     ax.annotate('', xy=(red_path["x"][150], red_path[1][150]), xytext=(path[0][140], path[1][140]),
#                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
#
#     # Black Trajectory (Inner Spiral)
#     black_path = model.simulate(0.35, 1.1, t_span=(0, 30))
#     ax.plot(black_path["x"], black_path["y"], color='black', linewidth=2)
#
#     # 3. Equilibrium Points (Brown dots)
#     # Stable interior point
#     ax.plot(0.33, 0.97, 'o', color='#8c6d31', markersize=10, markeredgecolor='white')
#     # Trivial points
#     ax.plot(1.0, 0, 'o', color='#8c6d31', markersize=8)
#     ax.plot(0, 0, 'o', color='#8c6d31', markersize=8)
#
#     # 4. Styling to match your image A/B
#     ax.set_xlim(0, 1.6)
#     ax.set_ylim(0, 2.0)
#     ax.set_xlabel('$x_1$', loc='right', fontsize=14)
#     ax.set_ylabel('$y_1$', loc='top', rotation=0, fontsize=14)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_xticks([0.5, 1.0, 1.5])
#     ax.set_yticks([0.5, 1.0, 1.5, 2.0])
#
#     plt.tight_layout()
#     plt.show()
#
# class plot_LVM:
#     def __init__(self,path):
#         self.path = path
#     def plot_PhaseSpace(self):
#         plt.plot(path["x"][0], path["y"][0], "bo")
#         plt.plot(path["x"], path["y"], color='blue', linewidth=1.5)
#         plt.xlim(0, 1.6)
#         plt.ylim(0, 2.0)
#         plt.xlabel('$x/prey$', loc='right', fontsize=14,color='b')
#         plt.ylabel('$y/predator$', loc='top', rotation=0, fontsize=14,color='b')
#         # plt.spines['top'].set_visible(False)
#         # plt.spines['right'].set_visible(False)
#         plt.xticks([0.5, 1.0, 1.5])
#         plt.yticks([0.5, 1.0, 1.5, 2.0])
#
#         plt.tight_layout()
#         plt.show()
#     def plot_traj(self,t_span):
#         fig, ax = plt.subplots(2,1)
#         ax[0].plot(path["x"], color='blue', linewidth=1.5)
#         ax[1].plot(path["y"], color='blue', linewidth=1.5)
#
#         ax[0].set_xlabel("time",fontsize=14,color='b')
#         ax[0].set_ylabel('$x/prey$', rotation=90, fontsize=14, color="b")
#
#         ax[1].set_xlabel("time",fontsize=14,color='b')
#         ax[1].set_ylabel('$y/predator$', rotation=90, fontsize=14,color="b")
#
#         plt.tight_layout()
#         plt.show()
#
#
#
#
#
# ######################## Plot Examples ###############################################
# ######################################################################################
# # Execution with stable parameters for spiraling behavior
# # model_inst = LV_M(alpha=1.0, beta=2.0, h=0.5, epsilon=0.5, m=0.3, H=0.05)
# # plot_journal_style(model_inst)
# model = LV_M(alpha=1.0, beta=2.0, h=0.5, epsilon=0.5, m=0.3, H=0.05)
# initial_cond = [1.5, 0.5]
# path = model.simulate(initial_cond[0], initial_cond[1], t_span=(0, 50))
# N_t=1000
# t_span = np.linspace(0, 50, N_t)
# plt_LV = plot_LVM(path)
# plt_LV.plot_PhaseSpace()
# plt_LV.plot_traj(t_span)