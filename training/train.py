import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import grad, jit
from models.get_model import init_model
from models.cvx_relu_mlp import CVX_ReLU_MLP
from utils.model_utils import optimal_weights_transform
from utils.opt_utils import get_optimizer
from optimizers.admm import admm
from time import perf_counter
from utils.metric_utils import get_model_performance
from utils.train_utils import get_batch

def train(Xtr, ytr, Xtst, ytst, model_params, opt_params, task):

  ######## SETUP ########

  if task not in ['regression', 'classification']:
     raise ValueError('This task is not supported!')
  
  ntr = ytr.shape[0]

  n_epoch = opt_params['n_epoch']
  batch_size = opt_params['batch_size']
  iters_in_epoch = jnp.ceil(ytr.shape[0]/batch_size).astype(jnp.int32)
  max_iters = jnp.round(iters_in_epoch*n_epoch).astype(jnp.int32)  
  
  # List to collect passes through data for plotting
  data_passes = []
  
  if task == 'classification':
     perf_log = {'train_loss': [], 'train_acc': [],
              'test_loss': [],'test_acc': []}
  else:
     perf_log = {'train_loss': [], 'train_acc': []}
  
  
  time_log = {'iteration_times': [], 'total_time': 0}

  seed = opt_params['seed']

  key, subkey = jax.random.split(seed)
  
  ytr_rs = ytr.reshape(ytr.shape[0], 1)
  ytst_rs = ytst.reshape(ytst.shape[0], 1)

  ytr_one_hot = jax.nn.one_hot(ytr, 10)

  ytst_one_hot = jax.nn.one_hot(ytst, 10)

  # Get model and initialization
  params, model, loss = init_model(model_params, Xtr[0,:], subkey)

  preds = model.apply(params, Xtr)
  
  # Get initial model performance
  data_passes.append(0)
  time_log['iteration_times'].append(0)
  perf_log = get_model_performance(perf_log, model, params, Xtr, Xtst, ytr, ytst, task)

  # Get optimizer and initial opt_state
  opt = get_optimizer(opt_params)
  opt_state = opt.optimizer.init(params)

  @jit
  def get_grad(params, data_batch, data_labels):
     return grad(loss)(params, data_batch, data_labels)

  ######## TRAINING LOOPS ########
  
  ####### LOOP FOR GRADIENT OPTIMIZERS #######
  if opt_params['optimizer'] in ['Adam', 'AdamW', 'DAdapt-AdamW', 'SGD', 
  'Shampoo', 'Yogi']:
    t_epoch = 0
    i, iter_counter = 0, 0 
    while i<= n_epoch-1:

      # Take one step of optimizer
      start = perf_counter()

      key, subkey = jrn.split(key)
      Idx = get_batch(subkey, ntr, batch_size)
      grads = get_grad(params, Xtr[Idx, :], ytr_one_hot[Idx,:])
      params, opt_state = opt.step(grads, params, opt_state)
      iter_counter+=1
    
      t_epoch+= perf_counter() - start
      
      if iter_counter % iters_in_epoch == 0:
        perf_log = get_model_performance(perf_log, model, params, Xtr, Xtst, ytr, ytst, task)
        time_log['iteration_times'].append(t_epoch)
        i+=1
        data_passes.append(i)
        t_epoch = 0
  
  
  # Get total time
  time_log['total_time'] = jnp.sum(jnp.array(time_log['iteration_times']))
  
  return model, params, perf_log, time_log, data_passes

