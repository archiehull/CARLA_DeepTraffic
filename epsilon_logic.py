import random
import numpy as np
import time
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.1

while True:
    r = np.random.random()
    if r > epsilon:
        print (f"{r:_>7.2f}{epsilon:_>7.2f}:Exploiting")

    else:
        print (f"{r:_>7.2f}{epsilon:_>7.2f}:Exploring")

    time.sleep(0.001)

    # Decrease epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
    
    else:  
        break