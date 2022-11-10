import pandas as pd
import numpy as np

def to_csv(fn="data/20200428_Imperative_Communities.xlsx", save_fn="data/2020_communities.csv"):
    pd.read_excel(fn, skiprows=3).to_csv(save_fn)

if __name__ == "__main__":
    to_csv()