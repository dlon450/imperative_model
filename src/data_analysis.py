import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from prepare_data import get_social_utilities

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def get_z(x):
    z = np.zeros((x.shape))
    for row, idx in enumerate(first_nonzero(x, 1)):
        if idx != -1:
            z[row][idx] = 1
    return z

def plot_results(results_fn, N, people_out_of_poverty, total_initial_investment, money_received_over_time, money_spent_over_time, needs_over_time):

    x = np.arange(1, N+1)
    plt.subplot(221)
    plt.plot(x, np.cumsum(people_out_of_poverty))
    plt.xlabel('Planning year')
    plt.ylabel('Number of people')
    plt.title('People lifted out of material poverty')

    plt.subplot(222)
    for need, y in needs_over_time.items():
        plt.plot(x, y, label=need)
    plt.legend(loc="upper right")
    plt.xlabel('Planning year')
    plt.ylabel('Number of people')
    plt.title('People with needs provided')

    plt.subplot(223)
    plt.plot(np.arange(1, N+2), total_initial_investment + np.cumsum(np.insert(money_received_over_time, 0, 0.)) \
        - np.cumsum(np.insert(money_spent_over_time, -1, 0.)), label='Funds remaining')
    plt.plot(np.arange(1, N+2), np.insert(money_received_over_time, 0, 0.), label='Cashflow received')
    plt.legend(loc="upper right")
    plt.xlabel('Planning year')
    plt.ylabel('$ (USD)')
    plt.title('Change in funds and cashflow over planning period')

    plt.subplot(224)
    plt.bar(np.arange(1, N+1), money_spent_over_time, label='Funds remaining')
    plt.xlabel('Planning year')
    plt.ylabel('$ (USD)')
    plt.title('Investments throughout planning period')
    plt.savefig(os.path.join('results', f'graphs_{results_fn[:results_fn.rfind(".")]}.png'))
    plt.show()

    # plt.plot(np.arange(1, N+1), 100 * np.cumsum(money_received_over_time) * 0.13 / 1e8)
    # plt.xlabel('Planning year')
    # plt.ylabel('Return (%)')
    # plt.show()

if __name__ == "__main__":

    ######################### CHANGE HERE ###########################

    communities_fn = os.path.join("example", "2020_communities.csv")
    results_fn = "x0.8_155436569.58.csv"

    alphas = np.arange(0., 1.1, 1.)
    total_initial_investment = 1e8
    timeframe = 5 # years
    timeframe_stepsize_in_months = 12
    investment_per_family = 6500
    discount_rate = 0.09
    interest_rate = 0.06
    repayment_period_in_months = 60

    ######################## DO NOT CHANGE ##########################

    needs = ["% of Population without Adequate Housing", "% of Population without Electricity", "% of Population without Water and Sanitation Services", "% of Households without access to Internet (Based on people without refrigerator)", "% of People without acces to quality Medical services (Municipality)"]
    communities_df = pd.read_csv(communities_fn).dropna()#.sample(frac=0.001, random_state=2)

    # filter out households that cannot meet monthly requirement
    repayment = interest_rate * investment_per_family / (1 - (1 + interest_rate) ** - (repayment_period_in_months / 12)) / 12 # monthly household/family repayment
    repayment_period_timesteps = repayment_period_in_months / timeframe_stepsize_in_months
    communities_df = communities_df[communities_df["Average monthly Income per Household (USD/m)"] >= 3 * repayment]
    J = len(communities_df)

    proportions_communities = communities_df[needs].to_numpy()
    households_communities = communities_df["Number of Households"].to_numpy()
    people_per_community = communities_df['Average Number of Inhabitants per Household'].to_numpy() * households_communities

    social_utilities = get_social_utilities(communities_df, proportions_communities, households_communities)
    cash_flows = np.array([repayment * h for h in households_communities]) * timeframe_stepsize_in_months

    x = pd.read_csv(os.path.join("results", results_fn), skiprows=1, header=None).to_numpy()
    z = get_z(x)

    # social_utility_over_time = np.dot(social_utilities, z)
    people_out_of_poverty = np.dot(people_per_community, z)
    money_spent_over_time = np.dot(investment_per_family*households_communities, z)
    money_received_over_time = np.dot((1+interest_rate)*cash_flows, x)
    needs_simplified = ["Housing", "Electricity", "Water/Sanitation", "Internet", "Municipality"]
    needs_over_time = {needs_simplified[i]: np.cumsum(np.dot(communities_df[need].to_numpy() * people_per_community, z)) for i, need in enumerate(needs)}

    plt.rcParams["figure.figsize"] = (25,15)
    plot_results(results_fn, x.shape[1], people_out_of_poverty, total_initial_investment, money_received_over_time, money_spent_over_time, needs_over_time)

    print(x)