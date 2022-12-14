import gurobipy as gp
from gurobipy import GRB 
import os
import time
import numpy as np

def solve_lp_prime(J, N, S, C, rd, ri, M, a, rpt, save=True):
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
    rd : float
        Capital growth rate (i.e., discount rate).
    ri : float
        Interest rate for cash flows.
    M : float
        Total initial investment for month 1.
    a : (J, 1) np.array of floats
        Cashflows per timestep for each community j.
    rpt : float
        Repayment period in same unit as timesteps.
    save : bool
        Indicate to save solution in 'results' directory.

    Returns
    -------
    S_prime : float
        Optimal objective (social utility) for LP'.
    N_prime : int
        First timestep where all communities have been reached.
    covering : bool
        Indicate if solution covers all communities.
    '''
    st = time.time()

    model = gp.Model("LP'")
    x = model.addVars(J, N, vtype=GRB.BINARY, name='x')
    z = model.addVars(J, N, vtype=GRB.BINARY, name='z')

    print('Setting objective...')
    model.setObjective(total_social_utility(S, z, rd, N) + x.sum() / J, GRB.MAXIMIZE)

    set_budget(model, x, z, C, M, N, ri, a)
    set_stationarity(model, J, x, z, N, rpt)
    
    ct = time.time()
    print("Elapsed time:", ct - st, "seconds")
    print('Solving...')
    S_prime, x_prime, _ = solve(model, J, N)
    print("Elapsed Time:", time.time() - ct, "seconds")

    covering, N_prime = get_first_covering_timestep(x_prime)

    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savetxt(os.path.join('results', f'x_prime_{S_prime:.2f}.csv'), x_prime.astype(int), delimiter=',', header=f'x_prime with objective {S_prime}')

    return S_prime - x_prime.sum() / J - 1, N_prime, covering

def solve_lp_alpha(J, N, S, C, rd, ri, M, a, S_prime, covering, alpha, rpt, save=True):
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
    rd : float
        Capital growth rate (i.e., discount rate).
    ri : float
        Interest rate for cash flows.
    M : float
        Total initial investment for month 1.
    a : (J, 1) np.array of floats
        Cashflows per timestep for each community j.
    S_prime : float
        Optimal objective (social utility) for LP'
    covering : bool
        Indicate if solution covers all communities.
    alpha : float
        Minimum proportion of S_prime to be achieved in LP
    rpt : float
        Repayment period in same unit as timesteps.
    save : bool
        Indicate to save solution in 'results' directory.

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
    model.setObjective(gp.quicksum( a @ x.select('*', n) / (1+rd)**n for n in range(N) ), GRB.MAXIMIZE)

    set_budget(model, x, z, C, M, N, ri, a)
    set_stationarity(model, J, x, z, N, rpt)

    print("Setting social utility constraint...")
    model.addConstr(total_social_utility(S, z, rd, N) >= alpha*S_prime)

    if covering:
        print("Setting covering constraint...")
        model.addConstr(z.sum('*', '*') == J)

    ct = time.time()
    print("Elapsed time:", ct - st, "seconds")
    print('Solving...')
    npv, x_best, z_best = solve(model, J, N)
    print("Elapsed Time:", time.time() - ct, "seconds")
    s_best = sum([S.dot(z_best[:,i]) / (1+rd)**i for i in range(N)])

    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savetxt(os.path.join('results', f'x{alpha:.1f}_{npv:.2f}.csv'), x_best.astype(int), delimiter=',', \
            header=f'x with alpha={alpha} and objective={npv} and s={s_best}')

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
            <= M + gp.quicksum( a @ x.select('*', t) for t in range(n) ), f"Budget_{n}")
    
def set_stationarity(model, J, x, z, N, rpt):
    print("\nSet stationarity constraints...")
    for j in range(J):
        expr = z.sum(j, "*")
        model.addConstr(expr <= 1)
        model.addConstr(x.sum(j, "*") <= rpt*expr)
        for n in range(N):
            model.addConstr(x[j, n] <= gp.quicksum(z[j, t] for t in range(n + 1)))

def total_social_utility(S, z, r, N):
    return gp.quicksum( S @ z.select('*', n) / (1+r)**n for n in range(N) )

def solve(m, J, N):

    # m.Params.MIPGap = 0.005   # 0.5%
    m.Params.TimeLimit = 3 * 3600  # 3 hour

    m.optimize()    
    print('Obj: %g' % m.objVal)
    x_ = m.getAttr("X", m.getVars()[:J*N])
    x_ = np.array(x_).reshape((J, N))
    z_ = m.getAttr("X", m.getVars()[J*N:])
    z_ = np.array(z_).reshape((J, N))
    return m.objVal, x_, z_

def get_first_covering_timestep(x_):
    try:
        return True, np.min(np.where(x_.all(axis=0))[0])
    except ValueError:
        return False, x_.shape[1]