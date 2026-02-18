import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import numpyro
import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import HalfCauchy
from numpyro.distributions import InverseGamma
from numpyro.distributions import Normal
import numpy as np


print("---------------------------- Model Define -------------------------")
############################## Horse-shoe dist ################################
def _sample_reg_horseshoe(tau: float, c_sq: float, shape: tuple[int, ...],name = "betaH"):

    lamb = numpyro.sample(name+"_λ", HalfCauchy(1.0), sample_shape=shape)
    lamb_squiggle = jnp.sqrt(c_sq) * lamb / jnp.sqrt(c_sq + tau**2 * lamb**2)
    betaH = numpyro.sample(
        name,
        Normal(jnp.zeros_like(lamb_squiggle), jnp.sqrt(lamb_squiggle**2 * tau**2)),
    )
    return betaH
############################## Horse-shoe Model ################################
def MultiTargetMultiEquation_HSModel(xx, yy):
    n_targets = yy.shape[1]

    n_IDs = xx.shape[0]
    n_features = xx.shape[1]
    n_obs = xx.shape[2]

    slab_shape_nu = 4
    slab_shape_s = 2
    noise_hyper_lambda = 1
    sparsity_coef_tau0 = 0.1
    print(f"n_features = {n_features}")
    # sample the horseshoe hyperparameters.
    τ_μ = numpyro.sample("τ_μ", HalfCauchy(sparsity_coef_tau0),sample_shape =(n_IDs,n_features,n_targets) )
    c_sq_μ = numpyro.sample("c_sq_μ",InverseGamma(slab_shape_nu / 2, slab_shape_nu / 2 * slab_shape_s**2),sample_shape =(n_IDs,n_features,n_targets))
    μ_coef = _sample_reg_horseshoe(τ_μ, c_sq_μ,( n_IDs,n_features,n_targets),"μ_coef")
    print(f"μ_coef.shape = {μ_coef.shape}")
    σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(1.),sample_shape =(n_IDs,n_features,n_targets))


    with numpyro.plate("plate_targets", n_targets):
      with numpyro.plate("plate_features", n_features):
        with numpyro.plate("plate_Indv", n_IDs):
          coef = numpyro.sample("coef", dist.Normal(μ_coef, σ_coef))

    y_est = jnp.einsum('ifo,ift->ito', xx, coef)
    # for each target we should consider different noise
    noise = numpyro.sample("noise", dist.HalfNormal(1),sample_shape=(n_targets,))
    noise = jnp.expand_dims(jnp.expand_dims(noise, axis=1), axis=0)
    noise = jnp.broadcast_to(noise, (n_IDs,n_targets,n_obs))

    numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)


################################# Flat Bayesian Horseshoe Model ############################
# def pre_Flat(xx, yy):
#     n_ind, n_coef, n_obs = xx.shape
#     n_ind, n_eqs, n_obs =yy.shape
#     x_out = np.reshape(xx, (n_coef, n_ind * n_obs))
#     y_out = np.reshape(yy, (n_eqs, n_ind * n_obs))
#     return x_out, y_out
def pre_Flat(xx, yy):
    n_indv, n_coef, n_obs = xx.shape
    n_indv, n_eq, n_obs = yy.shape
    xx_out = []
    yy_out = []

    for coef_i in range(n_coef):
        # Take the data for the current coefficient across all individuals and observations
        xx_coef_data = xx[:, coef_i, :] # Shape (n_indv, n_obs)
        # Reshape to (n_indv * n_obs,) and append to xx_out
        xx_out.append(xx_coef_data.reshape(-1))

    for eq_i in range(n_eq):
        # Take the data for the current equation across all individuals and observations
        yy_eq_data = yy[:, eq_i, :] # Shape (n_indv, n_obs)
        # Reshape to (n_indv * n_obs,) and append to yy_out
        yy_out.append(yy_eq_data.reshape(-1))

    # Stack the lists to get arrays of shape (n_coef, n_indv * n_obs) and (n_eq, n_indv * n_obs)
    xx_out = np.stack(xx_out, axis=0)
    yy_out = np.stack(yy_out, axis=0)

    print(xx_out.shape)
    print(yy_out.shape)

    return xx_out, yy_out

def Flat_HSModel(xx, yy):
    n_targets = yy.shape[0]

    n_features = xx.shape[0]
    n_obs = xx.shape[1]

    slab_shape_nu = 4
    slab_shape_s = 2
    noise_hyper_lambda = 1
    sparsity_coef_tau0 = 0.1
    print(f"n_features = {n_features}")
    # sample the horseshoe hyperparameters.
    τ_μ = numpyro.sample("τ_μ", HalfCauchy(sparsity_coef_tau0))
    c_sq_μ = numpyro.sample("c_sq_μ",InverseGamma(slab_shape_nu / 2, slab_shape_nu / 2 * slab_shape_s**2))
    coef = _sample_reg_horseshoe(τ_μ, c_sq_μ,( n_features,n_targets),"coef")
    print(f"coef = {coef.shape}")
    print(f"xx = {xx.shape}")

    y_est = jnp.einsum('fo,ft->to', xx, coef)
    print(f"y_est = {y_est.shape}")
    # for each target we should consider different noise
    noise = numpyro.sample("noise", dist.HalfNormal(1),sample_shape=(n_targets,))
    noise = jnp.expand_dims(noise, axis=1)
    noise = jnp.broadcast_to(noise, (n_targets,n_obs))
    print(f"noise = {noise.shape}")

    numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)




#################################################################
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import HalfCauchy, InverseGamma, Normal
import jax.numpy as jnp
# Assuming gen_param and other necessary imports are available in the execution environment



############################## Horse-shoe Model (with Student's t Likelihood) ################################
def TStudent_BHModel(xx, yy):
    n_targets = yy.shape[1]

    n_IDs = xx.shape[0]
    n_features = xx.shape[1]
    n_obs = xx.shape[2]

    slab_shape_nu = 4
    slab_shape_s = 2
    sparsity_coef_tau0 = 0.1
    print(f"n_features = {n_features}")

    # --- NEW: Degrees of Freedom (nu) for the Student's t-distribution ---
    # Sampling nu from a wide prior (e.g., Gamma or HalfNormal)
    # A smaller nu (e.g., < 10) indicates heavier tails and stronger robustness to outliers.
    # We constrain it to be >= 1 for numerical stability.
    df_nu = numpyro.sample("df_nu", dist.Gamma(2., 0.1))  # Mean of 20, but with flexibility
    # Ensure df_nu >= 1 for the StudentT distribution
    df_nu = jnp.clip(df_nu, 1.0, 100.)

    # sample the horseshoe hyperparameters.
    τ_μ = numpyro.sample("τ_μ", HalfCauchy(sparsity_coef_tau0), sample_shape=(n_IDs, n_features, n_targets))
    c_sq_μ = numpyro.sample("c_sq_μ", InverseGamma(slab_shape_nu / 2, slab_shape_nu / 2 * slab_shape_s ** 2),
                            sample_shape=(n_IDs, n_features, n_targets))
    μ_coef = _sample_reg_horseshoe(τ_μ, c_sq_μ, (n_IDs, n_features, n_targets), "μ_coef")
    print(f"μ_coef.shape = {μ_coef.shape}")
    σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(1.), sample_shape=(n_IDs, n_features, n_targets))

    with numpyro.plate("plate_targets", n_targets):
        with numpyro.plate("plate_features", n_features):
            with numpyro.plate("plate_Indv", n_IDs):
                coef = numpyro.sample("coef", dist.Normal(μ_coef, σ_coef))

    y_est = jnp.einsum('ifo,ift->ito', xx, coef)
    # for each target we should consider different scale (sigma)
    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(1), sample_shape=(n_targets,))
    noise_scale = jnp.expand_dims(jnp.expand_dims(noise_scale, axis=1), axis=0)
    noise_scale = jnp.broadcast_to(noise_scale, (n_IDs, n_targets, n_obs))

    # --- CRITICAL CHANGE: Student's t-distribution for the Likelihood ---
    # dist.StudentT(df, loc, scale)
    # df = degrees of freedom (df_nu), loc = mean (y_est), scale = noise_scale
    numpyro.sample("obs", dist.StudentT(df_nu, y_est, noise_scale), obs=yy)













    ################################### Normal Model ########################
    def MultiTargetMultiEquation_Normal(xx, yy):
        n_targets = yy.shape[1]

        n_IDs = xx.shape[0]
        n_features = xx.shape[1]
        n_obs = xx.shape[2]

        μ_coef = numpyro.sample("μ_coef", dist.Normal(0, 10.), sample_shape=(n_IDs, n_features, n_targets))
        σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(10.), sample_shape=(n_IDs, n_features, n_targets))

        with numpyro.plate("plate_targets", n_targets):
            with numpyro.plate("plate_features", n_features):
                with numpyro.plate("plate_Indv", n_IDs):
                    coef = numpyro.sample("coef", dist.Normal(μ_coef, σ_coef))

        y_est = jnp.einsum('ifo,ift->ito', xx, coef)
        # for each target we should consider different noise
        noise = numpyro.sample("noise", dist.HalfNormal(1), sample_shape=(n_targets,))
        noise = jnp.expand_dims(jnp.expand_dims(noise, axis=1), axis=0)
        noise = jnp.broadcast_to(noise, (n_IDs, n_targets, n_obs))

        numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)

########################## BH Model both μ , σ ############################
############################## Horse-shoe dist ################################
def _sample_reg_halfhorseshoe(tau: float, c_sq: float, shape: tuple[int, ...],name = "betaH"):

    lamb = numpyro.sample(name+"_λ", HalfCauchy(1.0), sample_shape=shape)
    lamb_squiggle = jnp.sqrt(c_sq) * lamb / jnp.sqrt(c_sq + tau**2 * lamb**2)
    betaH = numpyro.sample(
        name,
        dist.HalfNormal(jnp.sqrt(lamb_squiggle**2 * tau**2)),sample_shape = shape
    )
    return betaH
############################## Horse-shoe Model ################################
def Double_HSModel(xx, yy):
    n_targets = yy.shape[1]

    n_IDs = xx.shape[0]
    n_features = xx.shape[1]
    n_obs = xx.shape[2]

    slab_shape_nu = 4
    slab_shape_s = 2
    noise_hyper_lambda = 1
    sparsity_coef_tau0 = 0.1
    print(f"n_features = {n_features}")
    # sample the horseshoe hyperparameters.
    #
    with numpyro.plate("plate_targets", n_targets):
      with numpyro.plate("plate_features", n_features):
        with numpyro.plate("plate_Indv", n_IDs):
            τ_μ = numpyro.sample("τ_μ", HalfCauchy(sparsity_coef_tau0) )
            c_sq_μ = numpyro.sample("c_sq_μ",InverseGamma(slab_shape_nu / 2, slab_shape_nu / 2 * slab_shape_s**2))
            μ_coef = _sample_reg_horseshoe(τ_μ, c_sq_μ,(),"μ_coef")
            print(f"μ_coef.shape = {μ_coef.shape}")
            σ_coef = _sample_reg_halfhorseshoe(τ_μ, c_sq_μ,(),"σ_coef")
            print(f"μ_coef.shape = {μ_coef.shape}")
            print(f"σ_coef.shape = {σ_coef.shape}")
            coef = numpyro.sample("coef", dist.Normal(μ_coef, σ_coef))

    print(f"coef.shape = {coef.shape}")
    y_est = jnp.einsum('ifo,ift->ito', xx, coef)
    # for each target we should consider different noise
    noise = numpyro.sample("noise", dist.HalfNormal(1),sample_shape=(n_targets,))
    noise = jnp.expand_dims(jnp.expand_dims(noise, axis=1), axis=0)
    noise = jnp.broadcast_to(noise, (n_IDs,n_targets,n_obs))

    numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)


