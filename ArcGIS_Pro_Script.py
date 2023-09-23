import pandas as pd

X = pd.read_csv("!Arcgis.csv")

X['not deprived'] = X['not deprived']/X['All households']*100
X['deprived in one'] = X['deprived in one']/X['All households']*100
X['deprived in two'] = X['deprived in two']/X['All households']*100
X['deprived in three'] = X['deprived in three']/X['All households']*100
X['deprived in four'] = X['deprived in four']/X['All households']*100
X['L1 L2 and L3'] = X['L1 L2 and L3']/X['Sum']*100
X['L4 L5 and L6'] = X['L4 L5 and L6']/X['Sum']*100
X['L7'] = X['L7']/X['Sum']*100
X['L8 and L9'] = X['L8 and L9']/X['Sum']*100
X['L10 and L11'] = X['L10 and L11']/X['Sum']*100
X['L12'] = X['L12']/X['Sum']*100
X['L13'] = X['L13']/X['Sum']*100
X['L14.1 and L14.2'] = X['L14.1 and L14.2']/X['Sum']*100
X['L15'] = X['L15']/X['Sum']*100

X.to_csv("!Arc.csv")