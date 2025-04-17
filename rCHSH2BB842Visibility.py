from __future__ import print_function
from ncpol2sdpa import *
import pickle
from math import *
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import time
import qutip as qtp
def get_ith_digit_base_b(n, i, b):
    if b < 2:
        raise ValueError("Base must be greater than or equal to 2.")
    
    # Convert the number to base b and store the digits as a string
    base_str = ''
    while n > 0:
        base_str = str(n % b) + base_str
        n //= b
    
    # Pad with zeros to make sure we have 4 digits
    base_str = base_str.zfill(4)
    
    # Since the rightmost digit is the 0th digit, we can index from the end (-i - 1)
    if i < 0 or i >= len(base_str):
        raise ValueError(f"Invalid digit index {i}. Should be between 0 and {len(base_str)-1}.")
    
    return int(base_str[-i - 1])

def score_constraintsCHSH2(Aops, Bops, v, etaBL):

    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]

    # Assume a pure two-qubit state of |00> + |11> form
    rho = (cos(pi/4)*qtp.ket('00') + sin(pi/4)*qtp.ket('11')).proj()

    # Define the projectors for each of the measurements of Alice and Bob
    a00 = 0.5*(id + sz)
    a01 = id - a00
    a10 = 0.5*(id + sx)
    a11 = id - a10
    b00 = 0.5*(id + cos(pi/4)*sz + sin(pi/4)*sx)
    b01 = id - b00
    b10 = 0.5*(id + cos(pi/4)*sx - sin(pi/4)*sz)
    b11 = id - b10

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]

    # Now collect the constraints subject to the inefficient detection distribution
    constraints = []
    # Constraints of form p(00|xy)
    for x in range(4):
        
        x1 = get_ith_digit_base_b(x, 0, 2)
        x2 = get_ith_digit_base_b(x, 1, 2)

        for a in range(3):
            
            a1 = get_ith_digit_base_b(a, 0, 2)
            a2 = get_ith_digit_base_b(a, 1, 2)
            # Marginal constraints
            constraints += [Aops[x][a] - (v * (rho*qtp.tensor(A_meas[x1][a1], id)).tr().real * (rho*qtp.tensor(A_meas[x2][a2], id)).tr().real + (1-v) * 0.25)]
            constraints += [Bops[x][a] - (v * (rho*qtp.tensor(id, B_meas[x1][a1])).tr().real * (rho*qtp.tensor(id, B_meas[x2][a2])).tr().real + (1-v) * 0.25)]    
            for y in range(4):
                
                y1 = get_ith_digit_base_b(y, 0, 2)
                y2 = get_ith_digit_base_b(y, 1, 2)
                    
                for b in range(3):
                    
                    b1 = get_ith_digit_base_b(b, 0, 2)
                    b2 = get_ith_digit_base_b(b, 1, 2)
                    
                    constraints += [Aops[x][a]*Bops[y][b] - (v * (rho*qtp.tensor(A_meas[x1][a1], B_meas[y1][b1])).tr().real * (rho*qtp.tensor(A_meas[x2][a2], B_meas[y2][b2])).tr().real + (1-v) * 0.0625)]

                for bl in range(4):
                    
                    b1 = get_ith_digit_base_b(bl, 0, 2)
                    b2 = get_ith_digit_base_b(bl, 1, 2)
                    
                    constraints += [Aops[x][a]*Bops[y+4][bl] - etaBL * (v * (rho*qtp.tensor(A_meas[x1][a1], B_meas[y1][b1])).tr().real * (rho*qtp.tensor(A_meas[x2][a2], B_meas[y2][b2])).tr().real + (1-v) * 0.0625)]
        
        for bl in range(4):
            a1 = get_ith_digit_base_b(bl, 0, 2)
            a2 = get_ith_digit_base_b(bl, 1, 2)
        # Marginal constraints
            constraints += [Bops[x+4][bl] - etaBL *(v * (rho*qtp.tensor(id, B_meas[x1][a1])).tr().real * (rho*qtp.tensor(id, B_meas[x2][a2])).tr().real + (1-v) * 0.25)]    
    return constraints
def score_constraintsBB842(Aops, Bops, v, etaBL):

    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]

    # Assume a pure two-qubit state of |00> + |11> form
    rho = (cos(pi/4)*qtp.ket('00') + sin(pi/4)*qtp.ket('11')).proj()

    # Define the projectors for each of the measurements of Alice and Bob
    a00 = 0.5*(id + sz)
    a01 = id - a00
    a10 = 0.5*(id + sx)
    a11 = id - a10
    b00 = 0.5*(id + cos(pi/4)*sz + sin(pi/4)*sx)
    b01 = id - b00
    b10 = 0.5*(id + cos(pi/4)*sx - sin(pi/4)*sz)
    b11 = id - b10

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]

    # Now collect the constraints subject to the inefficient detection distribution
    constraints = []
    # Constraints of form p(00|xy)
    for x in range(4):
        
        x1 = get_ith_digit_base_b(x, 0, 2)
        x2 = get_ith_digit_base_b(x, 1, 2)

        for a in range(3):
            
            a1 = get_ith_digit_base_b(a, 0, 2)
            a2 = get_ith_digit_base_b(a, 1, 2)
            # Marginal constraints
            constraints += [Aops[x][a] - (v * (rho*qtp.tensor(A_meas[x1][a1], id)).tr().real * (rho*qtp.tensor(A_meas[x2][a2], id)).tr().real + (1-v) * 0.25)]
            constraints += [Bops[x][a] - (v * (rho*qtp.tensor(id, B_meas[x1][a1])).tr().real * (rho*qtp.tensor(id, B_meas[x2][a2])).tr().real + (1-v) * 0.25)]    
            for y in range(4):
                
                y1 = get_ith_digit_base_b(y, 0, 2)
                y2 = get_ith_digit_base_b(y, 1, 2)
                    
                for b in range(3):
                    
                    b1 = get_ith_digit_base_b(b, 0, 2)
                    b2 = get_ith_digit_base_b(b, 1, 2)
                    
                    constraints += [Aops[x][a]*Bops[y][b] - (v * (rho*qtp.tensor(A_meas[x1][a1], B_meas[y1][b1])).tr().real * (rho*qtp.tensor(A_meas[x2][a2], B_meas[y2][b2])).tr().real + (1-v) * 0.0625)]

                for bl in range(4):
                    
                    b1 = get_ith_digit_base_b(bl, 0, 2)
                    b2 = get_ith_digit_base_b(bl, 1, 2)
                    
                    constraints += [Aops[x][a]*Bops[y+4][bl] - etaBL * (v * (rho*qtp.tensor(A_meas[x1][a1], A_meas[y1][b1])).tr().real * (rho*qtp.tensor(A_meas[x2][a2], A_meas[y2][b2])).tr().real + (1-v) * 0.0625)]
        
        for bl in range(4):
            a1 = get_ith_digit_base_b(bl, 0, 2)
            a2 = get_ith_digit_base_b(bl, 1, 2)
        # Marginal constraints
            constraints += [Bops[x+4][bl] - etaBL * (v * (rho*qtp.tensor(id, A_meas[x1][a1])).tr().real * (rho*qtp.tensor(id, A_meas[x2][a2])).tr().real + (1-v) * 0.25)]    
    return constraints

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.

    Completely modifiable!
    """

    monos = []
    A_config = [4,4,4,4]
    B_config = [4,4,4,4,5,5,5,5]
    A = [Ai for Ai in generate_measurements(A_config, 'A')]
    B = [Bj for Bj in generate_measurements(B_config, 'B')]
    Aflat = flatten(A)
    BSflat = flatten([B[0],B[1],B[2],B[3]])
    BLflat = flatten([B[4],B[5],B[6],B[7]])
    
    for a0 in Aflat:
            for b1 in BLflat: 
                monos += [a0*b1]
    return monos[:]
def check():
    A_config = [4,4,4,4]
    B_config = [4,4,4,4,5,5,5,5]
    A = [Ai for Ai in generate_measurements(A_config, 'A')]
    B = [Bj for Bj in generate_measurements(B_config, 'B')]
    print("B measurements:", B)
    # Set η_BL symbolically according to the original prescription.
    etaBL = B[4][0] + B[4][1] + B[4][2] + B[4][3]

    P = Probability([4,4,4,4], [4,4,4,4,5,5,5,5])
    obj = -etaBL
    subs = P.substitutions
    for i in range(4,8):
        for j in range(4,8):
            if i != j:
                for a in range(4):
                    for b in range(4):
                        subs[B[i][a] * B[j][b]] = B[j][b] * B[i][a]
    flag=0
    
    score_cons = score_constraintsBB842(A, B, 0.995, etaBL)
    sdpRelaxation = SdpRelaxation(P.get_all_operators(), verbose=1)
    sdpRelaxation.get_relaxation(1, substitutions=subs,momentequalities=score_cons[:],extramonomials=P.get_extra_monomials('AB'))
    sdpRelaxation.set_objective(obj)
    
    # Set up v values and initialize eta_values list
    v_values = np.linspace(0.995,0.999,11)
    eta_values = []
    flag=0
    for v in v_values:
        if flag==1:
            score_cons = score_constraintsBB842(A, B, v, etaBL)
            sdpRelaxation.process_constraints(momentequalities=score_cons[:])
        sdpRelaxation.solve(solver="mosek")
        print("V=",v,"Intermediate primal value:",abs(sdpRelaxation.primal))
        eta = abs(sdpRelaxation.primal)
        eta_values.append(eta)
        flag=1

    # Check that the lengths match
    assert len(v_values) == len(eta_values), "Length mismatch between v_values and eta_values"
    csv_filename = "BB842_results995.csv"
    with open(csv_filename, 'w') as f:
        # Write CSV header
        f.write("v,eta\n")
        # Write each (v, η) pair
        for v_val, eta_val in zip(v_values, eta_values):
            f.write(f"{v_val},{eta_val}\n")

    print(f"Results saved to {csv_filename}")
    # Plotting η vs. v
    plt.figure()
    plt.plot(v_values, eta_values, marker='o', linestyle='-')
    plt.xlabel('v')
    plt.ylabel('η')
    plt.title('Plot of η vs. v')
    plt.grid(True)
    plt.show()


check()
