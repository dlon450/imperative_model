import gurobipy
import pandas as pd
import numpy as np

from model import solve_lp_prime, solve_lp_alpha, topsis
from prepare_data import get_social_utilities

ALPHA = np.arange(0., 1.1, 1.)
TOTAL = 1e8 # initial investment
TIMEFRAME = 5 # years
INVESTMENT = 6500 # per family
GROWTH_RATE = 0.06

def optimize(fn):
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
    communities_df = pd.read_csv(fn).dropna().sample(frac=0.001, random_state=2)

    J = len(communities_df)

    ####### filter out households that cannot meet monthly requirement ##########

    needs_colname = [
        "% of Population without Adequate Housing",
        "% of Population without Electricity",
        "% of Population without Water and Sanitation Services",
        "% of Households without access to Internet (Based on people without refrigerator)",
        "% of People without acces to quality Medical services (Municipality)",
    ]

    proportions_communities = communities_df[needs_colname].to_numpy()
    households_communities = communities_df["Number of Households"].to_numpy()
    monthly_household_incomes = communities_df["Average monthly Income per Household (USD/m)"].to_numpy()

    social_utilities = get_social_utilities(communities_df, proportions_communities, households_communities)
    cash_flows = monthly_household_incomes * households_communities * 12 # yearly cash flow

    S_prime, N_ = solve_lp_prime(J=J, N=TIMEFRAME, S=social_utilities, C=INVESTMENT*households_communities, \
        r=GROWTH_RATE, M=TOTAL, a=cash_flows)
    
    N_ = TIMEFRAME ###### for now #######

    n = len(ALPHA)
    npv_all, s_all, x_all = np.zeros(n), np.zeros(n), np.zeros((n, J, N_))
    for i, alpha in enumerate(ALPHA):
        npv_all[i], s_all[i], x_all[i] = solve_lp_alpha(J, N=N_, S=social_utilities, C=INVESTMENT*households_communities, r=GROWTH_RATE, M=TOTAL, \
            a=cash_flows, S_prime=S_prime, alpha=alpha)

    i_ = topsis(npv_all, s_all)

    return npv_all[i_], s_all[i_], x_all[i_]

if __name__ == "__main__":
    fn = "data/2020_communities.csv"
    optimize(fn)