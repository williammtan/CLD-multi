import jax
import jax.numpy as jnp
import jax.random as jrn
from models.cvx_relu_mlp import CVX_ReLU_MLP
from optimizers.admm import admm
from utils.load_data import load_cifar, load_fmnist



training_X, training_y, test_X, test_y = load_fmnist()

n_classes = 10
P_S = 32
beta = 10**-3
rho = 0.1
seed = jrn.key(0)
model = CVX_ReLU_MLP(training_X, training_y, n_classes, P_S, beta, rho, seed)
model.init_model()
model.Xtst = test_X
model.ytst = test_y

admm_params = dict(rank = 10, beta = beta, gamma_ratio = 1, admm_iters = 5, pcg_iters = 10, check_opt = False)

cvx_weights, metrics,_ = admm(model, admm_params)

print(metrics['val_acc']) 