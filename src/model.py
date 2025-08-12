import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import numpyro
import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import HalfCauchy
from numpyro.distributions import InverseGamma
from numpyro.distributions import Normal
import seaborn as sns

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
    σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(10.),sample_shape =(n_IDs,n_features,n_targets))


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


################################### Normal Model ########################
def MultiTargetMultiEquation_Normal(xx, yy):
    n_targets = yy.shape[1]

    n_IDs = xx.shape[0]
    n_features = xx.shape[1]
    n_obs = xx.shape[2]

    μ_coef = numpyro.sample("μ_coef", dist.Normal(0,10.), sample_shape=(n_IDs, n_features, n_targets))
    σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(10.),sample_shape =(n_IDs,n_features,n_targets))


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