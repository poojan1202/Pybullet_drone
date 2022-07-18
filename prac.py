import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import pickle

env = HoverAviary(gui=False,record=False)
env = Monitor(env,'monitor_1507_1')

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=65000,
                             render=False)
checkpoint_callback = CheckpointCallback(save_freq=97500, save_path='./logs/',
                                         name_prefix='rl_model')
callback = CallbackList([checkpoint_callback, eval_callback])

#model = SAC('MlpPolicy',env,verbose=1,device='cuda')
policy_kwargs = dict(ortho_init=False,activation_fn=th.nn.ReLU,net_arch=[dict(pi=[512,1024, 512,256,128], vf=[512,1024,512,265,128])])

model = PPO('MlpPolicy',env,policy_kwargs=policy_kwargs,verbose=1,learning_rate=0.0001)

# model = PPO.load("ppo_hover_2203_02", env=env)

model.learn(3600000,callback=callback,reset_num_timesteps=False)

t = env.get_episode_rewards()
model.save("drone_model")
del model




file_name = "rewards_1507.pkl"
op_file = open(file_name,'wb')
pickle.dump(t, op_file)
op_file.close()

fi,a = plt.subplots()
a.plot(np.arange(len(t)),t)
plt.show()

#HoverAviary

#BaseSingleAgentAviary
#Line 86
#Line 227


