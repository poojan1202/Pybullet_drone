import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from gym_pybullet_drones.utils.Logger import Logger
env = HoverAviary(gui=True,record=False)

model = PPO.load("ppo_hover_2203_02", env=env,custom_objects={"learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,})
obs = env.reset()
rew = []
logger = Logger(logging_freq_hz=int(1),
                num_drones=1)
for i in range(10):
  done=False
  t_r = 0
  while not done:
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    t_r+=reward
    #print('env.THRUST2WEIGHT_RATIO',env.THRUST2WEIGHT_RATIO)
    #env.render()
    if i==9:
      logger.log(drone=0,
                 timestamp=i / env.SIM_FREQ,
                 state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                 control=np.zeros(12))
    if done:
      obs = env.reset()
      rew.append(t_r)

logger.plot()





t = np.arange(10)

fig,ax = plt.subplots()
ax.plot(t,rew)
plt.show()