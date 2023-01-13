import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

data = pd.read_excel('/kaggle/input/obd-dataset/TestData.xlsx')
data.head()

