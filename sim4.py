#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
import os

################################'Generate regressor X from an AR(1)'##############################
################################ Xt = α + βXt-1 + εt

T=100;  # sample size
df = pd.DataFrame(index=range(T), columns=['x', 'y'])
alpha = 1;    # intercept parameter
beta = 0.9;   # autoregressive parameter
sigma = 0.1;  # standard error of innovations
x1 = alpha/(1-beta) # define initial value for time series x

epsilon = sigma*np.random.randn(T); # generate a vector of T random normal
df.iloc[0,:] = [x1] # initialize x

for i in range(0, T-1):
    df.iloc[i+1,0] = alpha + beta * df.iloc[i,0] + epsilon[i+1] # generate X(t) recursively
    # Xt = α + βXt-1 + εt

################################ Generate Y ##############################
################################ # Yt = α + βYt-1 + γXt + εt + θεt-1

T_2=100 
alpha_y = 1
beta_y = 0.9
gamma = 0.9
theta = 0.5
sigma = 0.1
epsilon_y = sigma*np.random.randn(T_2,1)
y1 = 0
df['y'] = 0

for j in range(0, T-1):
    df.iloc[j+1,1] = alpha_y + beta_y * df.iloc[j,1] + gamma * df.iloc[j+1,0] + epsilon_y[j] + theta * epsilon_y[j-1]
    # Yt = α + βYt-1 + γXt + εt + θεt-1

print(df)

################################ Simulation of regression ##############################
################################ Yt = α + βYt-1 + γXt + εt + θεt-1
# Convert dataframe to numpy arrays
y = df['y'].values.astype(float)
x = df['x'].values.astype(float)


# Fit the quantile regression model
quant_reg_1 = sm.QuantReg(y, x).fit(q=0.1)
quant_reg_5 = sm.QuantReg(y, x).fit(q=0.5)
quant_reg_9 = sm.QuantReg(y, x).fit(q=0.9)

# Print the regression summary
print(quant_reg_1.summary())
print(quant_reg_5.summary())
print(quant_reg_9.summary())
#print(quant_reg_1.summary().as_latex())

################################ Quantile regression plot ##############################
################################ Yt = α + βYt-1 + γXt + εt + θεt-1
# Create a scatter plot of the data
plt.scatter(x, y, alpha=0.5)

# Plot the quantile regression lines at different quantile levels
x_sort = np.sort(x)
y_pred_1 = quant_reg_1.predict(x_sort)
y_pred_5 = quant_reg_5.predict(x_sort)
y_pred_9 = quant_reg_9.predict(x_sort)

plt.plot(x_sort, y_pred_1, color='red', label='Quantile level: 0.1')
plt.plot(x_sort, y_pred_5, color='green', label='Quantile level: 0.5')
plt.plot(x_sort, y_pred_9, color='blue', label='Quantile level: 0.9')

# Add a legend and labels to the plot
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quantile regression at different quantile levels')
plt.show()

# the coefficient increases at as the quantile increases; also t-statistics increases, indicating that the relationship 
# between the predictor variable (X) and the response variable (Y) is stronger at the upper end of the distribution. 
# Specifically, a one-unit increase in X has a larger effect on Y at the 0.9 quantile level compared to the 0.1 and 0.5 quantile levels.

################################ Autocorrelation of the residuals ##############################
residuals_1 = quant_reg_1.resid # Calculate the residuals
residuals_5 = quant_reg_5.resid # Calculate the residuals
residuals_9 = quant_reg_9.resid # Calculate the residuals

# Plot the autocorrelation function of the residuals for all three quantile regression models in one plot
fig, ax = plt.subplots(3, 1, figsize=(8, 10))
plot_acf(quant_reg_1.resid, ax=ax[0])
ax[0].set_title('Autocorrelation of residuals (quantile level = 0.1)')
plot_acf(quant_reg_5.resid, ax=ax[1])
ax[1].set_title('Autocorrelation of residuals (quantile level = 0.5)')
plot_acf(quant_reg_9.resid, ax=ax[2])
ax[2].set_title('Autocorrelation of residuals (quantile level = 0.9)')
plt.tight_layout()
plt.show()

#Autocorrelation fades away as the lags increase


################################ OLS estimation of the parameters ##############################
# Add lagged values of Y and epsilon
y_lag = np.roll(y, 1)
y_lag[0] = 0
epsilon_lag = np.roll(epsilon_y, 1)
epsilon_lag[0] = 0

# Construct the design matrix
X = np.column_stack((np.ones_like(x), y_lag, x, epsilon_lag))

#This line creates a new numpy array called X that is the design matrix for regression. 
# It is created by stacking columns of np.ones_like(x) (a vector of ones with the same shape as x), 
# y_lag, x, and epsilon_lag horizontally using np.column_stack(). 
# The first column of X is a vector of ones to represent the intercept term.

# Estimate the parameters
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Print the parameter estimates
print("alpha_hat = {:.4f}".format(beta_hat[0]))
print("beta_hat = {:.4f}".format(beta_hat[1]))
print("gamma_hat = {:.4f}".format(beta_hat[2]))
print("theta_hat = {:.4f}".format(beta_hat[3]))

#print(X)


################################ GLS estimation of the parameters ##############################
resid = y - X @ beta_hat# Compute the residuals

acf_resid, ci = sm.tsa.stattools.acf(resid, nlags=10, alpha=0.05) # Compute the ACF of the residuals

# Estimate the covariance parameters
sigma2 = np.var(resid) * (1 - acf_resid[1])
print(sigma2.shape, type(sigma2))
sigma2_array = np.full(T, sigma2)
rho = acf_resid[1] / (1 - acf_resid[1])
S = np.diag(sigma2_array) + rho * np.diag(np.sqrt(sigma2_array[:-1] * sigma2_array[1:]), k=1) + rho * np.diag(np.sqrt(sigma2_array[:-1] * sigma2_array[1:]), k=-1)

# Compute the GLS estimator of the coefficients
beta_gls = np.linalg.inv(X.T @ np.linalg.inv(S) @ X) @ X.T @ np.linalg.inv(S) @ y

# Print the parameter estimates
print("alpha_hat = {:.4f}".format(beta_gls[0]))
print("beta_hat = {:.4f}".format(beta_gls[1]))
print("gamma_hat = {:.4f}".format(beta_gls[2]))
print("theta_hat = {:.4f}".format(beta_gls[3]))