import pandas as pd

df = pd.read_parquet("data/processed/scores.parquet")
print(df.groupby("ticker")["quarter"].apply(list).to_string())
