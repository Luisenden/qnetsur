import numpy as np
import matplotlib.pyplot as plt
import netsquid as ns
from netsquid.qubits import ketstates as ks
import netsquid.qubits.qubitapi as qapi
import inspect
import netsquid.qubits.qstate as qstate
import netsquid.qubits.operators as ops

ns.set_qstate_formalism(ns.QFormalism.DM)  


## elementary link entanglements
fidelities_depol = []
fidelities_werner_theory = []
probs = np.linspace(0,1,100)

for prob in probs:
    q1, q2 = qapi.create_qubits(2, 'Q0')
    bell_state1 = qapi.assign_qstate([q1,q2], ks.b00)  

    p = 0.25 * prob
    qapi.apply_pauli_noise(q1, [1-3*p,p,p,p])
    fidelity = qapi.fidelity([q1, q2], ks.b00, squared=True)
    fidelities_depol.append(fidelity)
    w = 1-prob
    fidelities_werner_theory.append( (3*w + 1)/4 )

plt.plot(probs, fidelities_depol, label='Depolarized')
plt.plot(probs, fidelities_werner_theory, '-.', label='Werner Theory')
plt.xlabel(r'$p$, $w = 1-p$')
plt.title(r'Link-Level Fidelity')
plt.legend()
plt.show()


fidelities_depol = []
fidelities_werner = []
fidelities_werner_theory = []
alphas = np.linspace(0, 0.1, 100)
for alpha in alphas:
    prob = 4/3*alpha
    # DEPOLARIZE BELL STATES
    q1, q2 = qapi.create_qubits(2, 'Q1')
    q3, q4 = qapi.create_qubits(2, 'Q2')
    bell_state1 = qapi.assign_qstate([q1,q2], ks.b00)  
    bell_state2 = qapi.assign_qstate([q3,q4], ks.b00)

    # depolarize
    # p = 0.25 * prob
    # qapi.apply_pauli_noise(q1, [1-3*p,p,p,p])
    # qapi.apply_pauli_noise(q3, [1-3*p,p,p,p])
    qapi.depolarize(q1, prob)
    qapi.depolarize(q3, prob)

    # bell measurement
    qapi.operate([q2, q3], ops.CX)
    qapi.operate(q2, ops.H)
    outcome_q2, _ = qapi.measure(q2)
    outcome_q3, _ = qapi.measure(q3)
    if outcome_q2 == 1: # apply correction operators depending on outcome
        qapi.operate(q1, ops.X)
    if outcome_q3 == 1:
        qapi.operate(q4, ops.Z)

    fidel_comps = [qapi.fidelity([q1, q4], ks.bell_states[i], squared=True) for i in range(3)]
    fidelity_depol = max(fidel_comps) # determine the fidelity 
    fidelities_depol.append(fidelity_depol)
    
    # COMPARE TO WERNER STATES
    q1w, q2w = qapi.create_qubits(2, 'W')
    w = 1-prob
    bell_statew = qapi.assign_qstate([q1w,q2w], ks.b00).dm
    rhow = w**2*bell_statew + (1-w**2)*np.identity(4)/4
    #print('IS VALID', ns.qubits.dmutil.is_valid_dm(rhow))
    werner_state = qapi.assign_qstate([q1w,q2w], rhow)

    fidelity = qapi.fidelity([q1w, q2w], ks.b00, squared=True)  # determine the fidelity 
    fidel_theory = (3*w**2+1)/4
    fidelities_werner.append(fidelity)
    fidelities_werner_theory.append(fidel_theory)

plt.plot(probs, fidelities_depol, label='Depolarized')
plt.plot(probs, fidelities_werner, label='Werner')
plt.plot(probs, fidelities_werner_theory, '-.', label='Werner Theory')
plt.xlabel(r'$p$, $w = 1-p$')
plt.title(r'E2E Fidelity')
plt.legend()
plt.show()