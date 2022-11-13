import pandas as pd
import numpy as np
from model import solve_lp_prime, solve_lp_alpha, topsis
from prepare_data import get_social_utilities

def optimize(fn, needs_colnames, initial, alphas, investment, timeframe, discount, interest, repayment):
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
    communities_df = communities_df[communities_df["Average monthly Income per Household (USD/m)"] >= 3 * repayment]
    J = len(communities_df)

    proportions_communities = communities_df[needs_colnames].to_numpy()
    households_communities = communities_df["Number of Households"].to_numpy()

    social_utilities = get_social_utilities(communities_df, proportions_communities, households_communities)
    cash_flows = np.array([repayment for _ in range(J)])

    S_prime, N_, covering = solve_lp_prime(J=J, N=timeframe, S=social_utilities, C=investment*households_communities, \
        r1=discount, r2=interest, M=initial, a=cash_flows)
    
    N_ = timeframe ###### for now #######

    n = len(alphas)
    npv_all, s_all, x_all = np.zeros(n), np.zeros(n), np.zeros((n, J, N_))
    for i, alpha in enumerate(alphas):
        npv_all[i], s_all[i], x_all[i] = solve_lp_alpha(J, N=N_, S=social_utilities, C=timeframe*households_communities, \
            r1=discount, r2=interest, M=initial, a=cash_flows, S_prime=S_prime, covering=covering, alpha=alpha, save=True)

    return npv_all, s_all, x_all

if __name__ == "__main__":

    fn = "example/2020_communities.csv"
    needs = ["% of Population without Adequate Housing", "% of Population without Electricity", "% of Population without Water and Sanitation Services", "% of Households without access to Internet (Based on people without refrigerator)", "% of People without acces to quality Medical services (Municipality)"]
    
    alphas = np.arange(0., 1.1, 1.)
    total_initial_investment = 1e8
    timeframe = 5 # years
    timeframe_stepsize_in_months = 12
    investment_per_family = 6500
    discount_rate = 0.06
    interest_rate = 0.06
    min_repayment_per_timestep = investment_per_family / timeframe / timeframe_stepsize_in_months

    npv, s, x = optimize(fn, needs, alphas, total_initial_investment, timeframe, timeframe_stepsize_in_months, \
        investment_per_family, discount_rate, interest_rate, min_repayment_per_timestep)
    
    print(npv, s)