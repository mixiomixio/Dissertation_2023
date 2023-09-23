import pandas as pd

data = pd.read_csv("!Arc.csv")

data = data.iloc[:,7:]

data.corr().to_csv("corr.csv")