import pandas as pd

with open('data.pkl', 'rb') as f:
    t = pd.read_pickle(f)
    print(1)