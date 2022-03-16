import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt
import numpy as np

env = HoverAviary(gui=False,record=False)
env.reset()
step = 0
#model = DDPG('MlpPolicy',env,verbose=1,gamma=0.99)
model = DDPG.load("ddpg_hover", env=env)
model.learning_rate=0.001
model.learn(2400,eval_freq=10)
model.save("ddpg_hover")
del model

env = HoverAviary(gui=True,record=False)
model = DDPG.load("ddpg_hover", env=env)
obs = env.reset()
rew = []
for i in range(100):
  done=False
  t_r = 0
  while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    t_r+=reward
    #env.render()
    if done:
      obs = env.reset()
      rew.append(t_r)




t = np.arange(100)

fig,ax = plt.subplots()
ax.plot(t,rew)
plt.show()