import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np
from models.cvx_relu_mlp import CVX_ReLU_MLP
from optimizers.admm import admm
from utils.load_data import load_fmnist
from experiments.lr_experiment import lr_random_search, lr_grid_exp_fun
import os
import pickle


# manually change the following variables DATASET and MODEL
DATASET = 'fmnist' # change to 'food', 'imgnet171', 'imgnet'
MODEL = 'mlp' # change to 'mlp', 'cnn', 'gpt2'
OUTPUT_DIR = '/home/miria/Downloads/ZACH/results/' # can change to relative directory 


training_X, training_y, test_X, test_y = load_fmnist()
print(training_y.shape)


  
# Setup optimizers
opts = {'Adam': dict(optimizer='Adam', n_epoch=30, batch_size=10000), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=30, batch_size=10000), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=30, batch_size=10000),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=30, batch_size=10000),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=30, batch_size=10000)}

dadam_params = dict(optimizer='DAdapt-AdamW', n_epoch=30, batch_size=10000, lr = 10**0, gamma = 0)

# CRONOS
n_classes = 10
P_S = 64
beta = 10**-3
rho = 0.1
seed = jrn.key(0)
model = CVX_ReLU_MLP(training_X, training_y, n_classes, P_S, beta, rho, seed)
model.init_model()
model.Xtst = test_X
model.ytst = test_y
cronos_params = dict(rank = 10, beta = beta, gamma_ratio = 1, admm_iters = 10, pcg_iters = 30, check_opt = False)

print('Training model with CRONOS')

# Run twice to get compiled version 
for i in range(2):
        _ , metrics, _ = admm(model, cronos_params)
        if i == 1:
            print('Finished training with CRONOS')
    
train_peak = np.max(metrics['train_acc'])
test_peak = np.max(metrics['val_acc'])
print(f"Peak train accuracy: {train_peak}")
print(f"Peak test accuracy: {test_peak}")


# Parameters for random search
l, u = -5.5, -1.5
grid_size = 5
tuning_seed = 1
optimizer_metrics = {}

problem_data = dict(training_X=training_X, training_y=training_y, test_X=test_X, test_y=test_y)
model_params = dict(type = 'two_layer_mlp')
task = 'classification'

for opt in opts:
    opts[opt]['seed'] = jax.random.key(0)
    optimizer_metrics[opt] = lr_random_search(problem_data, model_params, opts[opt], 
                                                 task, l, u, grid_size, tuning_seed)
    print("Finished tuning" + " " + opt + "!" )
    metrics = optimizer_metrics[opt]
    train_peak = np.max(np.max(metrics['train_acc']))
    test_peak = np.max(np.max(metrics['test_acc']))
    print(f"Peak train accuracy: {train_peak}")
    print(f"Peak test accuracy: {test_peak}")

# DAdam
print("Running DAdam")
for i in range(2):
    dadam_params['seed'] = jax.random.key(0)
    lr = dadam_params['lr']
    optimizer_metrics['DAdam'] = lr_grid_exp_fun(problem_data, model_params, dadam_params,
                                               task, np.array([lr]))

# Create the subfolder path
model_dir = os.path.join(OUTPUT_DIR, MODEL)

# Create the subfolder if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Define the full path for the pickle file
# CHECK filename correctly defined #########################################################################
filename = f"{DATASET}_{MODEL}_multclass.pkl"
pickle_file_path = os.path.join(model_dir, filename)

# Save the pickle file to the specified directory
with open(pickle_file_path, 'wb') as handle:
      pickle.dump(optimizer_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)