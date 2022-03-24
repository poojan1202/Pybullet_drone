import numpy as np
x = 0.1
y = 0.1
z = 0.75
dis = np.array([x,y,z])
reward = -1 * np.linalg.norm(np.array([0, 0, 1])-dis)**2
reward = 2*reward
if z<1.5:
    #reward = reward+2*(np.e**(-10*abs(z-1)))
    reward = reward+5*(np.e**(-10*abs(z-1)))
    #reward = reward + 2*np.e**(-10*abs(x)) + 2*np.e**(-10*abs(y))
    reward = reward-100*(abs(x+y))

if x>2.5 or y>2.5 or z>2.5:
    maxi = max(x,y,z)
    reward = reward-np.e**(maxi/3)

a = reward
print(a)