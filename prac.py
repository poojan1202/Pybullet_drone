import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import torch as th
import numpy as np

env = HoverAviary(gui=False,record=False)
env = Monitor(env,'monitor_name')
env.reset()
step = 0
policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[512,512,256,128])
#model = DDPG('MlpPolicy',env,verbose=1,gamma=0.99)
#model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model = PPO.load("ppo_hover_1", env=env)
#model.learning_rate=0.001
model.learn(10000)
t = env.get_episode_rewards()
fi,a = plt.subplots()
a.plot(np.arange(len(t)),t)
plt.show()
model.save("ppo_hover_1")
del model

env = HoverAviary(gui=True,record=False)

model = PPO.load("ppo_hover_1", env=env)
obs = env.reset()
rew = []
for i in range(10):
  done=False
  t_r = 0
  while not done:
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    t_r+=reward
    #print('env.THRUST2WEIGHT_RATIO',env.THRUST2WEIGHT_RATIO)
    #env.render()
    if done:
      obs = env.reset()
      rew.append(t_r)





t = np.arange(10)

fig,ax = plt.subplots()
ax.plot(t,rew)
plt.show()