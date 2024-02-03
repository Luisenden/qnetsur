import itertools as iter
import numpy as np
import pandas as pd

beta = 0.2 # attenuation in fiber
c_loss = 1 # other system losses
Fr = 0.84 # min threshold fidelity

distances_l = [50, 15] # [km]
attempt_durations_l = [10 ** -4, 10 ** -3] # [s]

eta_l = 10 ** (-0.1 * beta * np.array(distances_l)) # transmissivity
d_l = 3 * c_loss * eta_l / (2*np.array(attempt_durations_l))


def D_H(w):
    e2ew = np.prod(w)
    F = (3*e2ew+1)/4
    return 1 + F*np.log2(F) + (1-F) * np.log2((1-F)/3) # yield of the so-called “hashing” protocol

def U_D(R, w): # R is a scalar and w is a 2-vector
    val = np.log(R*D_H(w))
    return 0 if np.isnan(val) else val
    
def Objective(R, w):
    return - 2*U_D(R, w), (R - d_l[1]*(1-w[1]))**2, (2*R - d_l[0]*(1-w[0]))**2, (4*Fr-1)/3 - np.prod(w)

Ri = np.linspace(0.01,200,100)
w_user = np.linspace(0.01,0.99,100)
w_server = np.linspace(0.01,0.99,100)

result = {}
vars = list(iter.product(Ri,w_server,w_user))
for R,w_server,w_user in vars:
    result['obj'], result['const1'], result['const2'], result['Thres'] \
        = Objective(R,[w_server,w_user])

df = pd.DataFrame.from_records(result)




