from ruamel.yaml import StringIO, YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dagger_go1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
from raisimGymTorch.algo.ppo.dagger import DaggerExpert, DaggerAgent, DaggerTrainer
import torch.nn as nn
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exptid", type = int, help='experiment id to prepend to the run')
parser.add_argument("--overwrite", action = 'store_true')
parser.add_argument("--debug", action = 'store_true')
parser.add_argument("--loadpth", type = str, default = None)
parser.add_argument("--loadid", type = str, default = None)
parser.add_argument("--gpu", type = int, default = 0)
# parser.add_argument("--name", type = str)
parser.add_argument("--ext_act", type = str, default='leakyRelu')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(args.loadpth + "/cfg.yaml", 'r'))
with open("cfg.yaml", 'r') as f:
    dagger_cfg = YAML().load(f)
# set seed
rng_seed = cfg['seed']
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

t_steps = dagger_cfg['environment']['history_len']
base_dims = cfg['environment']['baseDim']
n_futures = int(dagger_cfg['environment']['n_futures'])
cfg['environment']['n_futures'] = n_futures
prop_latent_dim = 8
geom_latent_dim = 1

activation_fn_map = {'none': None, 'tanh': nn.Tanh, 'leakyRelu': nn.LeakyReLU}
output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
small_init_flag = cfg['architecture']['small_init']
ext_activation_map = activation_fn_map[args.ext_act]

if args.debug:
    cfg['environment']['num_envs'] = 1
    cfg['environment']['num_threads'] = 1
    device = 'cpu'

    clip = True
else:
    device = 'cuda:{}'.format(args.gpu)

    clip = False

cfg['environment']['num_threads'] = dagger_cfg['environment']['num_threads']

cfg['environment']['history_len'] = t_steps
cfg['environment']['test'] = False
cfg['environment']['eval'] = False

class MyYAML(YAML):
    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()

yaml = MyYAML()   # or typ='safe'/'unsafe' etc

# create environment from the configuration file
env = VecEnv(dagger_go1.RaisimGymEnv(home_path + "/rsc", yaml.dump(cfg['environment'])), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts


priv_dim = ob_dim - base_dims * (t_steps + 1)

# save a few logs about the run
cfg['environment']['loadpth'] = args.loadpth
cfg['environment']['loadid'] = args.loadid


# save the configuration and other files
saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/third_phase/" + '{:04d}'.format(args.exptid),
                           save_items=[],
                           config = cfg, overwrite = args.overwrite)

expert_policy = DaggerExpert(args.loadpth, args.loadid, ob_dim, t_steps,
                             base_dims, env.obs_rms.mean.shape[0],
                             geomDim=int(cfg['environment']['geomDim']),
                             n_futures=n_futures, clip = clip)

student_mlp = ppo_module.MLP(cfg['architecture']['policy_net'],
                                        ext_activation_map,
                                        base_dims + prop_latent_dim + (n_futures+1)*geom_latent_dim,
                                        act_dim,
                                        output_activation_fn, 
                                        small_init_flag)

student_mlp.architecture.load_state_dict(expert_policy.policy.action_mlp.state_dict())

fname_mlp = saver.data_dir+"/mlp_"+str(args.loadid)+'.pt'
example_input = torch.rand(1, ob_dim - priv_dim + prop_latent_dim + (n_futures+1)*geom_latent_dim).cpu()
hlen = base_dims * t_steps

mlp_graph = torch.jit.trace(student_mlp.architecture.to(device), example_input[:, hlen:])
torch.jit.save(mlp_graph, fname_mlp)

student_mlp.to(device)

# prop_latent_encoder = ppo_module.StateHistoryEncoder(ext_activation_map, base_dims, t_steps,
#                                                      prop_latent_dim + (n_futures+1)*geom_latent_dim)

# actor = DaggerAgent(expert_policy,
#                     prop_latent_encoder,
#                     student_mlp, t_steps, base_dims, device, n_futures=n_futures)


# actor.save_deterministic_graph_temp(saver.data_dir+"/mlp_"+str(args.loadid)+'.pt', 
#                                     torch.rand(1, ob_dim - priv_dim + prop_latent_dim + (n_futures+1)*geom_latent_dim).cpu())

