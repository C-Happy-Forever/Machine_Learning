import pandas as pd
from ABSOLUTE_PATH import *
data = pd.read_table(open(default_path + '/100.dat'))
data.to_csv(default_path + '/100.csv')

