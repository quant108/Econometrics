import math
import numpy as np

#(a)

V = -2.488 + 0.954*1+0.914*1+0.070*50-0.069*(50/10)**2
q1 = (1-math.exp(V)/(1+math.exp(V)))*1*0.914

print(V, q1)

# (c)

q3 = (math.exp(0.914)-1)/(1+math.exp(V))

print(q3)
