import jax
import jax.numpy as jnp
import jax.random as jrn
from models.cvx_relu_mlp import CVX_ReLU_MLP
from optimizers.admm import admm
from utils.load_data import load_cifar, load_fmnist


training_X, training_y, test_X, test_y = load_fmnist()

jax.config.update("jax_enable_x64", True)

P_S = 32
beta = 10**-3
rho = 0.1
seed = jrn.key(0)
model = CVX_ReLU_MLP(training_X, training_y, P_S, beta, rho, seed)
model.init_model()
model.Xtst = test_X
model.ytst = test_y

def matvec_Fi(i, vec):
    return model.d_diags[:,i] * (model.X @ vec)

def matvec_F(vec):
  n = model.X.shape[0]
  out = jnp.zeros((n,))
  for i in range(model.P_S):
    out += matvec_Fi(i, vec[0,:,i] - vec[1,:,i])
  return out

def rmatvec_Fi(i, vec):
  return  model.X.T @ (model.d_diags[:,i] * vec)

def rmatvec_F(vec):
  n, d = model.X.shape
  out = jnp.zeros((2, d, model.P_S))
  for i in range(model.P_S):
    rFi_v = rmatvec_Fi(i,vec)
    out = out.at[0,:,i].set(rFi_v)
    out = out.at[1,:,i].set(-rFi_v)
    return out

def matvec_Gi(i, vec):
  return model.e_diags[:,i] * (model.X @ vec)

def matvec_G(vec):
        n, d = model.X.shape
        out = jnp.zeros((2,n, model.P_S))
        for i in range(model.P_S):
            out = out.at[0,:,i].set(matvec_Gi(i,vec[0,:,i]))
            out = out.at[1,:,i].set(matvec_Gi(i,vec[1,:,i]))
        return out

def rmatvec_Gi(i, vec):
  return model.X.T@(model.e_diags[:,i]*vec)

def rmatvec_G(vec):
        n, d = model.X.shape
        out = jnp.zeros((2, d,model.P_S))
        for i in range(model.P_S):
            out = out.at[0,:,i].set(rmatvec_Gi(i,vec[0,:,i]))
            out = out.at[1,:,i].set(rmatvec_Gi(i,vec[1,:,i]))
        return out

u = jrn.normal(jrn.key(0), (60000,))

print(rmatvec_F(u)-model.rmatvec_F(u.reshape(60000, 1)))