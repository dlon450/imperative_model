import pandas as pd
import numpy as np
import os
from model import solve_lp_prime, solve_lp_alpha, topsis
from prepare_data import get_social_utilities

def optimize(fn, needs_colnames, initial, alphas, investment, timeframe, discount, interest, repayment_period, stepsize, save=True):
    '''
    Formulate LP' and LP to find optimal strategy for investment allocation.

    Parameters
    ----------
    fn : str
        Filename for communities csv.

    Returns
    -------
    npv : float
        Optimal NPV achieved by our solution.
    s : float
        Optimal social utility achieved by our solution.
    x : (J, N) binary np.array
        Optimal solution/strategy for investment allocation.
    '''
    communities_df = pd.read_csv(fn).dropna()#.sample(frac=0.001, random_state=2)

    # filter out households that cannot meet monthly requirement
    repayment = investment_per_family / repayment_period
    repayment_period_timesteps = repayment_period / stepsize
    communities_df = communities_df[communities_df["Average monthly Income per Household (USD/m)"] >= 3 * repayment]
    J = len(communities_df)

    proportions_communities = communities_df[needs_colnames].to_numpy()
    households_communities = communities_df["Number of Households"].to_numpy()

    social_utilities = get_social_utilities(communities_df, proportions_communities, households_communities)
    cash_flows = np.array([repayment for _ in range(J)]) * stepsize

    S_prime, N_, covering = solve_lp_prime(J=J, N=timeframe, S=social_utilities, C=investment*households_communities, \
        rd=discount, ri=interest, M=initial, a=cash_flows, rpt=repayment_period_timesteps, save=save)
    
    # N_ = timeframe ###### for now #######

    n = len(alphas)
    npv_all, s_all, x_all = np.zeros(n), np.zeros(n), np.zeros((n, J, N_))
    for i, alpha in enumerate(alphas):
        npv_all[i], s_all[i], x_all[i] = solve_lp_alpha(J, N=int(N_), S=social_utilities, C=investment*households_communities, \
            rd=discount, ri=interest, M=initial, a=cash_flows, S_prime=S_prime, covering=covering, alpha=alpha, rpt=repayment_period_timesteps, save=save)

    return npv_all, s_all, x_all

if __name__ == "__main__":

    fn = os.path.join("example", "2020_communities.csv")
    needs = ["% of Population without Adequate Housing", "% of Population without Electricity", "% of Population without Water and Sanitation Services", "% of Households without access to Internet (Based on people without refrigerator)", "% of People without acces to quality Medical services (Municipality)"]
    
    alphas = np.arange(0., 1.1, 1.)
    total_initial_investment = 1e8
    timeframe = 10 # years
    timeframe_stepsize_in_months = 12
    investment_per_family = 6500
    discount_rate = 0.06
    interest_rate = 0.06
    repayment_period_in_months = 60

    npv, s, x = optimize(fn, needs, total_initial_investment, alphas, investment_per_family, timeframe, \
        discount_rate, interest_rate, repayment_period_in_months, timeframe_stepsize_in_months)
    
    print(npv, s)