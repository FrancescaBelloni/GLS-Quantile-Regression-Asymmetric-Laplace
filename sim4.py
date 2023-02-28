import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


T=100;  # sample size
df = pd.DataFrame(index=range(T+1), columns=['x', 'y'])
alpha = 1;    # intercept parameter
beta = 0.9;   # autoregressive parameter
sigma = 0.1;  # standard error of innovations
x1 = alpha/(1-beta) # define initial value for time series x

epsilon = sigma*np.random.randn(T,1); # generate a vector of T random normal
                                          # variables with variance sigma^2

df.iloc[0,:] = [x1, 0] # Place x1 in the first row first column of the DataFrame - column-row

for i in range(1, T+1):
    df.iloc[i,0] = alpha + beta * df.iloc[i-1,0] + epsilon[i-1,0] # generate x(t) recursively

# Simulation of regression 
T_2=100 
gamma = 0.9
delta = 0.5
sigma = 0.1
epsilon_y = sigma*np.random.randn(T_2,1)
epsilon_y_t_1 = sigma*np.random.randn(T_2,1)
y1 = 0

df.iloc[:,1] = y1,

for j in range(1, T+1):
    df.iloc[j,1] = df.iloc[j-1,1] + gamma * df.iloc[j,0] + epsilon_y[j-1,0] + delta * epsilon_y_t_1[j-1,0]

print(df)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# fit the model
model = smf.quantreg('y ~ x', df).fit(q=0.7)
 
# view model summary
print(model.summary())     

