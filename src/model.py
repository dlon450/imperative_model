import gurobipy as gp
from gurobipy import GRB 
import time
import numpy as np

def solve_lp_prime(J, N, S, C, r, M, a):
    '''
    Solve LP' and return S', x', N

    Parameters
    ----------
    J : int
        Total number of communities (rows) in csv file.
    N : int
        Total number of timesteps to be considered initially.
    S : (J, 1) np.array of floats
        Social utilities for each community j.
    C : (J, 1) np.array of floats
        Cost of investment for all households in community j.
    r : float
        Capital growth rate (i.e., discount rate).
    M : float
        Total initial investment for month 1.
    a : (J, 1) np.array of floats
        Cashflows per timestep for each community j.

    Returns
    -------
    S_prime : float
        Optimal objective (social utility) for LP'.
    N_prime : int
        First timestep where all communities have been reached.
    '''
    st = time.time()

    model = gp.Model("LP'")
    x = model.addVars(J, N, vtype=GRB.BINARY, name='x')
    z = model.addVars(J, N, vtype=GRB.BINARY, name='z')

    print('Setting objective...')
    model.setObjective(total_social_utility(S, x, r, N), GRB.MAXIMIZE)

    set_budget(model, x, z, C, M, N, r, a)
    set_stationarity(model, J, x, z, N)
    
    ct = time.time()
    print("Elapsed time:", ct - st, "seconds")
    print('Solving...')
    S_prime, x_prime = solve(model, J, N)
    print("Elapsed Time:", time.time() - ct, "seconds")

    N_prime = get_first_covering_timestep(x_prime)

    return S_prime, N_prime

def solve_lp_alpha(J, N, S, C, r, M, a, S_prime, alpha):
    '''
    Solve LP and return objective and solution

    Parameters
    ----------
    J : int
        Total number of communities (rows) in csv file.
    N : int
        Total number of timesteps to be considered initially.
    S : (J, 1) np.array of floats
        Social utilities for each community j.
    C : (J, 1) np.array of floats
        Cost of investment for all households in community j.
    r : float
        Capital growth rate (i.e., discount rate).
    M : float
        Total initial investment for month 1.
    a : (J, 1) np.array of floats
        Cashflows per timestep for each community j.
    S_prime : float
        Optimal objective (social utility) for LP'
    alpha : float
        Minimum proportion of S_prime to be achieved in LP

    Returns
    -------
    npv : float
        Optimal NPV (LP objective) achieved by our solution.
    s_best : float
        Social utility achieved by our solution.
    x_best : (J, N) binary np.array
        Optimal solution for investment allocation.
    '''
    st = time.time()

    model = gp.Model("LP_alpha")
    x = model.addVars(J, N, vtype=GRB.BINARY, name='x')
    z = model.addVars(J, N, vtype=GRB.BINARY, name='z')

    print("Setting Objective...")
    model.setObjective(gp.quicksum( a @ x.select('*',n) / (1+r)**n for n in range(N) ), GRB.MAXIMIZE)

    set_budget(model, x, z, C, M, N, r, a)
    set_stationarity(model, J, x, z, N)

    print("\nSetting social utility constraint...")
    model.addConstr(total_social_utility(S, x, r, N) >= alpha*S_prime)

    print("Setting covering constraint...")
    model.addConstr(z.sum('*', '*') == J)

    ct = time.time()
    print("Elapsed time:", ct - st, "seconds")
    print('Solving...')
    npv, x_best = solve(model, J, N)
    print("Elapsed Time:", time.time() - ct, "seconds")
    s_best = sum([S.dot(x_best[:,i]) / (1+r)**i for i in range(N)])

    return npv, s_best, x_best

def topsis():
    '''
    Return best from Pareto-Optimal
    '''
    pass

def set_budget(model, x, z, C, M, N, r, a):
    print("Setting budget constraints...", end=" ")
    model.addConstr(C @ z.select('*', 0) <= M, "Budget_0")
    for n in range(1, N):
        print(n, end=" ")
        model.addConstr(gp.quicksum( C @ z.select('*', t) for t in range(n + 1) ) \
            <= M + r*gp.quicksum( a @ x.select('*', t) for t in range(n) ), f"Budget_{n}")

def set_stationarity(model, J, x, z, N):
    print("\nSet stationarity constraints...")
    for j in range(J):
        model.addConstr(z[j,0] == x[j,0])
        for n in range(1, N):
            model.addConstr(x[j,n] - z[j,n] == x[j,n-1])

def total_social_utility(S, x, r, N):
    return gp.quicksum( S @ x.select('*',n) / (1+r)**n for n in range(N) )

def solve(m, J, N):
    m.optimize()

    # for v in m.getVars():
        # print('%s %g' % (v.varName, v.x))
    
    print('Obj: %g' % m.objVal)

    x_ = m.getAttr("X", m.getVars()[:J*N])
    x_ = np.array(x_).reshape((J, N))

    # z_ = m.getAttr("X", m.getVars()[J*N:])
    # z_ = np.array(z_).reshape((J, N))

    return m.objVal, x_

def get_first_covering_timestep(x_):
    return np.min(np.where(x_.all(axis=0))[0])