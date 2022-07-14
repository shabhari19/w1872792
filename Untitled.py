#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance==0.1.70


# In[2]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy.stats as si
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[3]:


initial_data = yf.download("SOL-GBP", start="2021-03-01", end="2022-03-01") #downloading data from yahoo finance


# In[4]:


initial_data.head()#showing the first 5 data information


# In[5]:


initial_data[['Adj Close']].head()


# In[6]:


data =  initial_data[['Adj Close']]
data.head()


# In[7]:


data.describe().round(2)   #showig data description


# In[8]:


data.plot(figsize=(10, 7), subplots=True) #plotting the movement of the stock


# In[9]:


log_return = np.log(initial_data['Adj Close'] / initial_data['Adj Close'].shift(1)) #calculating the volaltilty of the stock


# In[10]:


vol = log_return.std()    #daily volatility
print('The daily volatility is', round(vol*100,2), '%')


# In[11]:


vol = np.sqrt(252) * log_return.std()  #annualised volatility
print('The annualised volatility is', round(vol*100,2), '%')


# In[12]:


log_return.dropna(inplace=True) #volatility graph
log_return.plot(subplots=True, figsize=(10, 6))


# In[13]:


last_six_vol = round(data['Adj Close'][125:].pct_change().apply(lambda x: np.log(1+x)).std() * np.sqrt(252)*100, 2)
print('The last six months volatility is', last_six_vol, '%')   # last six months volatility


# In[15]:


first_six_vol = round(data['Adj Close'][:125].pct_change().apply(lambda x: np.log(1+x)).std() * np.sqrt(252)*100, 2)
print('The first six months volatility is', first_six_vol, '%') # first six months volatility   


# # QUESTION 2:-
# 
# methodology

# # BINOMIAL TREE
# This involves using an iterative apporaoch utilizing multiple steps to value option prices. The model is based on the assumption that there is a risk neutral enivronment, the interest rates are constant and the price of underlying asset takes two values; goes up or goes down

# In[16]:


S = initial_data['Adj Close'][-1]
print('The spot price is', round(S,2))  # getting the spot price


# In[17]:


S0 = 73.98          # spot stock price
K = 85              # strike price
T = 1/12             # time to maturity 
r =  0.0169            # risk free rate 
sig = 1.1864         # annaulised volatility
N =  5               # number of time steps of the tree
payoff =  "call"     # payoff


# In[18]:


dT = float(T) / N                             # Delta t
u = np.exp(sig * np.sqrt(dT))                 # up probability factor
d = 1.0 / u   


# In[19]:


S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[20]:


S


# In[21]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # probability of the price of underlying asset going up
q = 1.0 - p           # probability of the price of underlying asset going down
p


# In[22]:


q


# In[23]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# In[24]:


# for European call Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[25]:


print('European ' + payoff, str( V[0,0]))


# # Monte Carlo Simulation
# This is amethod used to calculate the value of an option by simulationg the random walk and generating a sequence of random numbers showing the future value of stock based on which we calcuale the future value of the option.

# In[26]:


def mcs_simulation_np(p):
    M = p
    I = p
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t]) 
    return S


# In[27]:


T = 1/12
r = 0.0169
sigma = 1.1864
S0 = 73.98
K = 85


# In[28]:


S = mcs_simulation_np(10000)    #number of simulations


# In[29]:


S = np.transpose(S)
S


# In[30]:


import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=S[:,-1], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-preiod values')


# In[31]:


p = np.mean(np.maximum(K - S[:,-1],0))
print('European call', str(p))


# # Greeks
# Greeks measure the sensitivity of the price of a derivative with respect to a certain risk factor

# # Delta
# Delta, measures the rate of change of the theoritical option value with rspect to changes in the price of the underlying asset. Delta is the first derivative of the value of the option with respect to the price of the underlying asset.

# In[32]:


def delta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    elif payoff == "put":
        delta =  - np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)
    
    return delta


# In[33]:


delta(73.98,85,1/12,0.0169,0.54,1.1864, 'call') # value of delta


# In[34]:


S = np.linspace(65,90, 21) #plotting the 3d graph of delta
T = np.linspace(0.2,0.083, 21)
Delta = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Delta[i,j] = delta(S[j], 85, T[i],0.6169,0.54, 1.1864, 'call')


# In[35]:


fig = plt.figure(figsize=(10, 6)) #dimension of the graph
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Delta, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Delta')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[36]:


S = np.linspace(65,90,11)  #plotting the 2d graph 
Delta_Call = np.zeros((len(S),1))
Delta_Put = np.zeros((len(S),1))
for i in range(len(S)):
    Delta_Call [i] = delta(S[i], 85, 1/12, 0.0169, 0.54, 1.1864, 'call')


# In[37]:


fig = plt.figure()
plt.plot(S, Delta_Call, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Delta')
plt.legend(['Delta for Call'])


# In[38]:


d = delta(73.98,85,1/12,0.0169,0.54,1.1864, 'call')
print('The value of Delta is', d.round(4),'.','If the stock price increase 1 dollar, then the value of the option will increase $', d.round(4), '.')


# # GAMMA
# Gamma measures the rate of change in delta with respect to changes in the underlying price.Gamma is the option price second derivative with respect to its underlying price.

# In[39]:


def gamma(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    gamma = np.exp(- q * T) * si.norm.pdf(d1, 0.0, 1.0) / (vol * S * np.sqrt(T))
    
    return gamma


# In[40]:


gamma(73.98,85,1/12,0.0169,0.54,1.1864, "call")


# In[41]:


S = np.linspace(65,90,21) #plotting the 3d graph
T = np.linspace(0.2,0.083, 21)
Gamma = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Gamma[i,j] = gamma(S[j], 85, T[i], 0.0169, 0.54, 1.1864, 'call')


# In[42]:


fig = plt.figure(figsize=(10, 6)) #dimension of the graph
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Gamma, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Gamma')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[43]:


S = np.linspace(65,90,21) #plotting the 2d graph
Gamma = np.zeros((len(S),1))
for i in range(len(S)):
    Gamma [i] = gamma(S[i], 85, 1/12,0.6169,0.54, 1.1864, 'call')


# In[44]:


fig = plt.figure()
plt.plot(S, Gamma, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Gamma')
plt.title('Gamma')
plt.legend(['Gamma for Call'])


# In[45]:


d = gamma(73.98,85,1/12,0.0169,0.54,1.1864, 'call')
print('The value of Gamma is', d.round(4),'.','If the stock price increase 1 dollar, then the value of the option will increase $', d.round(4), '.')


# # RHO
# Rho measures the sensitivity of the option to changes in the interest rates, it is the derivative of the option value with respect to the risk free interest rate.

# In[46]:


def rho(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        rho =  K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        rho = - K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return rho


# In[47]:


rho(73.98,85,1/12,0.6169,0.54,1.1864, 'call') #value of rho


# In[48]:


S = np.linspace(65,90, 21) #plotting the 3d graph of rho
T = np.linspace(0.2,0.083, 21)
Rho = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Rho[i,j] = rho(S[j], 85, T[i],0.6169,0.54,1.1864, 'call')


# In[49]:


fig = plt.figure(figsize=(10, 6))  #dimension of the graph
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Rho, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Rho')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[50]:


r = np.linspace(0.2,0.083,11)  #plotting the 2d graph
Rho_Call = np.zeros((len(r),1))
for i in range(len(r)):
    Rho_Call [i] = rho(73.98,85, 1/12, r[i], 0.54, 1.1864, 'call')


# In[51]:


fig = plt.figure()
plt.plot(r, Rho_Call, '-')
plt.grid()
plt.xlabel('Interest Rate')
plt.ylabel('Rho')
plt.title('Rho')
plt.legend(['Rho for Call'])


# In[52]:


r = rho(73.98,85, 1/12,0.6169,0.54,1.1864, 'call')
print('The value of Rho is', r.round(4),'.','If the interest rate increases 1%, then the value of the option will increase $', r.round(4)*0.01, '.')


# # VEGA
# Vega measures the sensitivity of the option to changes in volatility. Vega is the derivative of the option value with to the volatility of the underlying asset.

# In[53]:


def vega(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    vega = S * np.sqrt(T) * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0)
    
    return vega


# In[54]:


vega(73.98,85,1/12,0.6169,0.54,1.1864, 'call')  #value of vega


# In[55]:


S = np.linspace(65,90, 21) #plotting the 3d graph
T = np.linspace(0.1,0.083, 31)
Vega = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Vega[i,j] = vega(S[j], 77, T[i], 0.07, 0.6169,1.1864, 'call')


# In[56]:


fig = plt.figure(figsize=(10, 6)) #dimensins of the graph
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Vega, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Vega')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[57]:


vol = np.linspace(65,90,11)    #plotting the 2d graph
Vega = np.zeros((len(vol),1))
for i in range(len(vol)):
    Vega [i] = vega(73.98,85,1/12,0.6169,0.54, vol[i], 'call')


# In[60]:


fig = plt.figure()
plt.plot(vol, Vega, '-')
plt.grid()
plt.xlabel('Volatility')
plt.ylabel('Vega')
plt.title('Vega')
plt.legend(['Vega for Call'])


# In[59]:


v = vega(73.98,85,1/12,0.6169,0.54,1.1864, 'call')
print('The value of Vega is', v.round(4),'.','If the volatility increases 1%, then the value of the option will increase $', v.round(4)*0.01, '.')


# # THETA
# Theta, measures the sensitivity of the value of the derivative to the change in the time to expiration

# In[61]:


def theta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.2 * vol ** 0.083) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(-d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta


# In[62]:


theta(73.98,85,1/12,0.6169,0.54,1.1864, 'call') #value of theta 


# In[63]:


S = np.linspace(65,90, 21)  #plotting the 3d graph
T = np.linspace(0.2,0.083, 21)
Theta = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Theta[i,j] = theta(S[j],85, T[i],0.6169,0.54,1.1864, 'call')


# In[64]:


fig = plt.figure(figsize=(10, 6))    #dimensions of thr graph
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Theta, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Theta')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[65]:


T = np.linspace(0.2,0.083,12)   #plotting the 2d graph
Theta_Call = np.zeros((len(T),1))
for i in range(len(T)):
    Theta_Call [i] = theta(73.98,85, T[i],0.6169,0.54,1.1864, 'call')


# In[66]:


fig = plt.figure()
plt.plot(T, Theta_Call, '-')
plt.grid()
plt.xlabel('Time to Expiry')
plt.ylabel('Theta')
plt.title('Theta')
plt.legend(['Theta for Call'])


# In[67]:


t = theta(73.98,85,1/12,0.6169,0.54,1.1864, 'call')
print('The value of Theta is', t.round(4),'.','If the time increases 1 year, then the value of the option will increase $', t.round(4)*0.01, '.')

