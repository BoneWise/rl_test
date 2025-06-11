import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# import matplotlib_fontja # 如果不需要日文字体，可以暂时注释掉
import seaborn as sns

from scipy.optimize import minimize
import pingouin as pg

import requests
import io

#2.2
print(2.2)
def func_TaskSetting(pA=0.8):
    df=pd.DataFrame({
        'block': [1]*80+[2]*80,
        'trial': [i+1 for i in range(160)],
        'outA': np.r_[np.random.choice([1, 0], 80, p=[pA, 1-pA], replace=True), np.random.choice([1, 0], 80, p=[1-pA, pA], replace=True)],
        'outB': np.r_[np.random.choice([1, 0], 80, p=[1-pA, pA], replace=True), np.random.choice([1, 0], 80, p=[pA, 1-pA], replace=True)]
    })
    return df

np.random.seed(123)
setting=func_TaskSetting()

# 打印出数据框的前10行
print(setting.head(10))

print(setting.rename(columns={'outA': 'A', 'outB': 'B'}).groupby('block')[['A', 'B']].mean())

#2.3
print(2.3)
fig=plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.5)

ax1=fig.add_subplot(2, 1, 1)
ax1.scatter(setting['trial'], setting['outA'], color='red')
ax1.axvline(81, linestyle='--', color='black')
ax1.set_xlabel('trial')
ax1.set_ylabel('outA')
ax1.xaxis.set_major_locator(MultipleLocator(40))
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.set_title('select A')

ax2=fig.add_subplot(2, 1, 2)
ax2.scatter(setting['trial'], setting['outB'], color='blue')
ax2.axvline(81, linestyle='--', color='black')
ax2.set_xlabel('trial')
ax2.set_ylabel('outB')
ax2.xaxis.set_major_locator(MultipleLocator(40))
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.set_title('select B')
fig.show()

#3.1
print(3.1)
alpha=0.2
Q=0.4
R=1
Q=Q+alpha*(R-Q)
Q

alphas=np.arange(6)*0.2
for i in alphas:
    alpha=i
    Q=0.4
    R=1
    Q=Q+alpha*(R-Q)
    print(f'alpha: {alpha:.1f}, update Q: {Q}')

alpha=0.2
Q=[None]*5
Q[0]=0.5
R=[1, 1, 0, 1, 1]
for t in range(4):
    Q[t+1]=Q[t]+alpha*(R[t]-Q[t])
    Q

[[i+1, Q[i]] for i in range(5)]

#3.2
print(3.2)
beta=5
Q_A=1.0
Q_B=0.8
P_A=1/(1+np.exp(-beta*(Q_A-Q_B)))
P_B=1-P_A

print(f'P(A): {P_A}')
print(f'P(B): {P_B}')

#3.3
P_A=0.8
c=[]
if P_A>np.random.rand():
    c+=['A']
else:
    c+=['B']
c

np.random.seed(123)
c=[None]*100
P_A=0.8
for i in range(100):
    if P_A>np.random.rand():
        c[i]=1
    else:
        c[i]=2

np.c_[np.unique(c, return_counts=True)].T

#4.1
print(4.1)
def func_DataGeneration(param, setting):
    alpha=param[0]
    beta=param[1]

    outA=setting['outA'].values
    outB=setting['outB'].values
    trials=setting['trial'].values
    Ntrial=len(trials)

    c=[None]*Ntrial
    r=[None]*Ntrial
    Q_A=[None]*Ntrial
    Q_B=[None]*Ntrial
    p_A=[None]*Ntrial
    RPE=[None]*Ntrial

    Q_A[0]=0.5
    Q_B[0]=0.5

    for t in range(Ntrial):
        p_A[t]=np.exp(beta*Q_A[t])/(np.exp(beta*Q_A[t])+np.exp(beta*Q_B[t]))

        if p_A[t]>np.random.rand():
            c[t]=1
            r[t]=outA[t]
            RPE[t]=r[t]-Q_A[t]
        else:
            c[t]=2
            r[t]=outB[t]
            RPE[t]=r[t]-Q_B[t]

        if t<Ntrial-1:
            if c[t]==1:
                Q_A[t+1]=Q_A[t]+alpha*RPE[t]
                Q_B[t+1]=Q_B[t]

            if c[t]==2:
                Q_B[t+1]=Q_B[t]+alpha*RPE[t]
                Q_A[t+1]=Q_A[t]

    df=pd.DataFrame({
        'trial': trials,
        'QA': Q_A,
        'QB': Q_B,
        'choice': c,
        'reward': r,
        'pA': p_A,
        'pB': [1-p for p in p_A],
        'RPE': RPE
    })

    return df

#4.2
print(4.2)
np.random.seed(123)
tmp_param=[0.2, 5]
tmp_setting=func_TaskSetting(0.8)

res=func_DataGeneration(tmp_param, tmp_setting)
print(res)

fig=plt.figure(figsize=(6, 6))
ax=fig.add_subplot(1, 1, 1)
ax.plot(res['trial'], res['QA'], color='red')
ax.plot(res['trial'], res['QB'], color='blue')
ax.axvline(81, linestyle='--', color='black')
ax.set_xlabel('trial')
ax.set_ylabel('Q')
ax.set_xlim(-10, 170)
ax.set_ylim(-0.1, 1.1)
ax.xaxis.set_major_locator(MultipleLocator(40))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_title(f'learning rate{tmp_param[0]}, inverse temperature{tmp_param[1]}')
fig.show()

#4.3
print(4.3)
Nsim=100

res100=pd.DataFrame()
np.random.seed(123)
tmp_param=[0.2, 5]

for i in range(Nsim):
    tmp_setting=func_TaskSetting()
    res=func_DataGeneration(param=tmp_param, setting=tmp_setting)
    res=pd.merge(res, tmp_setting, left_on='trial', right_on='trial', how='left')
    res.insert(0, 'id', i)

    res100=pd.concat([res100, res])

res100.shape

res100.head(10)


fig=plt.figure(figsize=(6, 6))
ax=fig.add_subplot(1, 1, 1)
ax.plot(np.arange(160)+1, res100.groupby('trial')['QA'].mean().values, color='red')
ax.plot(np.arange(160)+1, res100.groupby('trial')['QB'].mean().values, color='blue')
ax.axvline(81, linestyle='--', color='black')
ax.set_xlabel('trial')
ax.set_ylabel('Q')
ax.set_xlim(-10, 170)
ax.set_ylim(-0.1, 1.1)
fig.show()

#5.1
print(5.1)
url = 'https://x.gd/Ne4xj'
res = requests.get(url).content
data = pd.read_csv(io.StringIO(res.decode('utf-8')), header=0)
data.head(10)

#5.3
print(5.3)
def func_LogLik(param, data):
    alpha=param[0]
    beta=param[1]

    trials=data['trial']
    Ntrial=len(trials)
    c=data['choice']
    r=data['reward']

    Q_A=[None]*Ntrial
    Q_B=[None]*Ntrial
    p_A=[None]*Ntrial
    RPE=[None]*Ntrial

    Q_A[0]=0.5
    Q_B[0]=0.5

    logLik=0

    for t in range(Ntrial):
        p_A[t]=np.exp(beta*Q_A[t])/(np.exp(beta*Q_A[t])+np.exp(beta*Q_B[t]))

        if c[t]==1:
            RPE[t]=r[t]-Q_A[t]
            logLik=logLik+np.log(max(p_A[t], 1e-14))  #logの安定化のため1e-14を下限にしている

        if c[t]==2:
            RPE[t]=r[t]-Q_B[t]
            logLik=logLik+np.log(max(1-p_A[t], 1e-14))  #logの安定化のため1e-14を下限にしている

        if t<Ntrial-1:
            if c[t]==1:
                Q_A[t+1]=Q_A[t]+alpha*RPE[t]
                Q_B[t+1]=Q_B[t]

            if c[t]==2:
                Q_B[t+1]=Q_B[t]+alpha*RPE[t]
                Q_A[t+1]=Q_A[t]

    return logLik

data1=data.loc[data['id']==1, :]
betas=np.arange(0.05, 8, 0.05)

LogLik_1=[]

for beta in betas:
    param=[0.2, beta]
    tmp=func_LogLik(param=param, data=data1)
    LogLik_1=LogLik_1+[tmp]

df=pd.DataFrame({
    'beta': betas,
    'LogLik': LogLik_1
})

fig=plt.figure(figsize=(6, 6))
ax=fig.add_subplot(1, 1, 1)
ax.plot(df['beta'], df['LogLik'])
ax.set_xlabel('inverse temperature')
ax.set_ylabel('log likelihood ')
fig.show()

k=np.argmax(LogLik_1)
betas[k]