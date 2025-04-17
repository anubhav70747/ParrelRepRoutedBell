from __future__ import print_function
from ncpol2sdpa import *
import pickle
from math import *
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import time
import qutip as qtp


def score_constraintsCHSH(Aops, Bops, v, etaBL):
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    rho = (cos(pi/4) * qtp.ket('00') + sin(pi/4) * qtp.ket('11')).proj()
    a00 = 0.5 * (id + sz)
    a01 = id - a00
    a10 = 0.5 * (id + sx)
    a11 = id - a10
    b00 = 0.5 * (id + cos(pi/4) * sz + sin(pi/4) * sx)
    b01 = id - b00
    b10 = 0.5 * (id + cos(pi/4) * sx - sin(pi/4) * sz)
    b11 = id - b10
    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]
    constraints = []
    constraints += [Bops[2][0] + Bops[2][1] - etaBL]
    constraints += [Bops[3][0] + Bops[3][1] - etaBL]
    for x in range(2):
        for y in range(2):
            constraints += [Aops[x][0] * Bops[y][0] - (v * (rho * qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + (1 - v) * 0.25)]
            constraints += [Aops[x][0] * Bops[y + 2][0] - etaBL * (v * (rho * qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + (1 - v) * 0.25)]
            constraints += [Aops[x][0] * Bops[y + 2][1] - etaBL * (v * (rho * qtp.tensor(A_meas[x][0], B_meas[y][1])).tr().real + (1 - v) * 0.25)]
    
    # Marginal constraints
    constraints += [Aops[0][0] - (v * (rho * qtp.tensor(A_meas[0][0], qtp.qeye(2))).tr().real + (1 - v) * 0.5)]
    constraints += [Aops[1][0] - (v * (rho * qtp.tensor(A_meas[1][0], qtp.qeye(2))).tr().real + (1 - v) * 0.5)]
    
    constraints += [Bops[1][0] - (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[1][0])).tr().real + (1 - v) * 0.5)]
    constraints += [Bops[0][0] - (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[0][0])).tr().real + (1 - v) * 0.5)]

    constraints += [Bops[2][0] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[0][0])).tr().real + (1 - v) * 0.5)]
    constraints += [Bops[2][1] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[0][1])).tr().real + (1 - v) * 0.5)]

    constraints += [Bops[3][0] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[1][0])).tr().real + (1 - v) * 0.5)]
    constraints += [Bops[3][1] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[1][1])).tr().real + (1 - v) * 0.5)]

    return constraints

def score_constraintsBB84(Aops, Bops, v, etaBL):
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    rho = (cos(pi/4) * qtp.ket('00') + sin(pi/4) * qtp.ket('11')).proj()
    a00 = 0.5 * (id + sz)
    a01 = id - a00
    a10 = 0.5 * (id + sx)
    a11 = id - a10
    b00 = 0.5 * (id + cos(pi/4) * sz + sin(pi/4) * sx)
    b01 = id - b00
    b10 = 0.5 * (id + cos(pi/4) * sx - sin(pi/4) * sz)
    b11 = id - b10
    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]
    constraints = []
    constraints += [Bops[2][0] + Bops[2][1] - etaBL]
    constraints += [Bops[3][0] + Bops[3][1] - etaBL]
    for x in range(2):
        for y in range(2):
            constraints += [Aops[x][0] * Bops[y][0] - (v * (rho * qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + (1 - v) * 0.25)]
            constraints += [Aops[x][0] * Bops[y + 2][0] - etaBL * (v * (rho * qtp.tensor(A_meas[x][0], A_meas[y][0])).tr().real + (1 - v) * 0.25)]
            constraints += [Aops[x][0] * Bops[y + 2][1] - etaBL * (v * (rho * qtp.tensor(A_meas[x][0], A_meas[y][1])).tr().real + (1 - v) * 0.25)]
    
    # Marginal constraints
    constraints += [Aops[0][0] - (v * (rho * qtp.tensor(A_meas[0][0], qtp.qeye(2))).tr().real + (1 - v) * 0.5)]
    constraints += [Aops[1][0] - (v * (rho * qtp.tensor(A_meas[1][0], qtp.qeye(2))).tr().real + (1 - v) * 0.5)]
    
    constraints += [Bops[1][0] - (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[1][0])).tr().real + (1 - v) * 0.5)]
    constraints += [Bops[0][0] - (v * (rho * qtp.tensor(qtp.qeye(2), B_meas[0][0])).tr().real + (1 - v) * 0.5)]

    constraints += [Bops[2][0] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), A_meas[0][0])).tr().real + (1 - v) * 0.5)]
    constraints += [Bops[2][1] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), A_meas[0][1])).tr().real + (1 - v) * 0.5)]

    constraints += [Bops[3][0] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), A_meas[1][0])).tr().real + (1 - v) * 0.5)]
    constraints += [Bops[3][1] - etaBL * (v * (rho * qtp.tensor(qtp.qeye(2), A_meas[1][1])).tr().real + (1 - v) * 0.5)]

    return constraints

def check():
    A_config = [2, 2]
    B_config = [2, 2, 3, 3]
    A = [Ai for Ai in generate_measurements(A_config, 'A')]
    B = [Bj for Bj in generate_measurements(B_config, 'B')]
    print("B measurements:", B)
    # Set η_BL symbolically according to the original prescription.
    etaBL = B[2][0] + B[2][1]

    P = Probability([2, 2], [2, 2, 3, 3])
    obj = -etaBL
    subs = P.substitutions
    for i in range(2, 4):
        for j in range(2, 4):
            if i != j:
                for a in range(2):
                    for b in range(2):
                        subs[B[i][a] * B[j][b]] = B[j][b] * B[i][a]
    score_cons = score_constraintsCHSH(A, B, 1/2**0.5, etaBL)
    sdpRelaxation = SdpRelaxation(P.get_all_operators(), verbose=1)
    sdpRelaxation.get_relaxation(3, substitutions=subs,momentequalities=score_cons[:])
    sdpRelaxation.set_objective(obj)
    
    # Set up v values and initialize eta_values list
    v_values = np.linspace(1/2**0.5,1,300)
    eta_values = []
    
    for v in v_values:
        score_cons = score_constraintsCHSH(A, B, v, etaBL)
        sdpRelaxation.process_constraints(momentequalities=score_cons[:])
        sdpRelaxation.solve(solver="mosek")
        print("Intermediate primal value:", abs(sdpRelaxation.primal))
        eta = abs(sdpRelaxation.primal)
        eta_values.append(eta)
    
    # Check that the lengths match
    assert len(v_values) == len(eta_values), "Length mismatch between v_values and eta_values"
    csv_filename = "CHSH_results.csv"
    with open(csv_filename, 'w') as f:
        # Write CSV header
        f.write("v,eta\n")
        # Write each (v, η) pair
        for v_val, eta_val in zip(v_values, eta_values):
            f.write(f"{v_val},{eta_val}\n")
    # Plotting η vs. v
    plt.figure()
    plt.plot(v_values, eta_values, marker='o', linestyle='-')
    plt.xlabel('v')
    plt.ylabel('η')
    plt.title('Plot of η vs. v')
    plt.grid(True)
    plt.show()


check()
