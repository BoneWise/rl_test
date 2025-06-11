import numpy as np

class Agent():
    def __init__(self, OOXX_Index, Epsilon, Alpha):
        self.index = OOXX_Index
        self.epsilon = Epsilon
        self.alpha = Alpha
        self.value = np.zeros((3,3,3,3,3,3,3,3,3))
        self.stored_Outcome = np.zeros(9).astype(np.int8)
    def reset(self):
        self.stored_Outcome = np.zeros(9).astype(np.int8)

    def move(self, State):
        Outcome = State.copy()
        available = np.where(Outcome == 0)[0]
        if np.random.binomial(1, self.epsilon):
            Outcome[np.random.choice(available)] = self.index
        else:
            temp_Value = np.zeros(len(available))
            for i in range(len(available)):
                temp_Outcome = Outcome.copy()
                temp_Outcome[available[i]] = self.index
                temp_Value[i] = self.value[tuple(temp_Outcome)]
            choose = np.argmax(temp_Value)
            Outcome[available[choose]] = self.index
        Error = self.value[tuple(Outcome)] - self.value[tuple(self.stored_Outcome)]
        self.value[tuple(self.stored_Outcome)] += self.alpha * Error
        self.stored_Outcome = Outcome.copy()
        return Outcome

def Judge(Outcome, OOXX_Index):
    Triple = np.repeat(OOXX_Index, 3)
    winner = 0
    if(Outcome[0:3]==Triple).all() or (Outcome[3:6]==Triple).all() or (Outcome[6:9]==Triple).all() :
        winner = OOXX_Index
    if (Outcome[0:7:3] == Triple).all() or (Outcome[1:8:3] == Triple).all() or (Outcome[2:9:3] == Triple).all():
        winner = OOXX_Index
    if (Outcome[0:9:4] == Triple).all() or (Outcome[2:7:2] == Triple).all():
        winner = OOXX_Index
    if 0 not in Outcome:
        winner = 3
    return winner

Agent1 = Agent(1, 0.1, 0.1)
Agent2 = Agent(2, 0.1, 0.1)
Trial = 30000
Winner = np.zeros(Trial)
for i in range(Trial):
    if i == 20000:
        Agent1.epsilon = 0
        Agent2.epsilon = 0
    Agent1.reset()
    Agent2.reset()
    winner = 0
    State = np.zeros(9).astype(np.int8)
    while winner == 0:
        Outcome = Agent1.move(State)
        winner = Judge(Outcome, 1)
        if winner == 1:
            Agent1.value[tuple(Outcome)] = 1
            Agent2.value[tuple(State)] = -1
        elif winner == 0:
            State = Agent2.move(Outcome)
            winner = Judge(State, 2)
            if winner == 2:
                Agent2.value[tuple(State)] = 1
                Agent1.value[tuple(Outcome)] = -1
    Winner[i] = winner


import matplotlib.pyplot as plt

# 根据结果计算胜率
step = 250 # 每隔250局游戏计算一次胜率
duration = 500 # 胜率根据前后共500局来计算
def Rate(Winner):
    Rate1 = np.zeros(int((Trial-duration)/step)+1) # Agent1 胜率
    Rate2 = np.zeros(int((Trial-duration)/step)+1) # Agent2 胜率
    Rate3 = np.zeros(int((Trial-duration)/step)+1) # 平局概率
    for i in range(len(Rate1)):
        Rate1[i] = np.sum(Winner[step*i:duration+step*i]==1)/duration
        Rate2[i] = np.sum(Winner[step*i:duration+step*i]==2)/duration
        Rate3[i] = np.sum(Winner[step*i:duration+step*i]==3)/duration
    return Rate1,Rate2,Rate3

Rate1,Rate2,Rate3=Rate(Winner)

fig,ax=plt.subplots(figsize=(16,9))
plt.plot(Rate1,linewidth=4,marker='.',markersize=20,color="#0071B7",label="Agent1")
plt.plot(Rate2,linewidth=4,marker='.',markersize=20,color="#DB2C2C",label="Agent2")
plt.plot(Rate3,linewidth=4,marker='.',markersize=20,color="#FAB70D",label="Draw")
plt.xticks(np.arange(0,121,40),np.arange(0,31+1,10),fontsize=30)
plt.yticks(np.arange(0,1.1,0.2),np.round(np.arange(0,1.1,0.2),2),fontsize=30)
plt.xlabel("Trials(x1k)",fontsize=30)
plt.ylabel("Winning Rate",fontsize=30)
plt.legend(loc="best",fontsize=25)
plt.tick_params(width=4,length=10)
ax.spines[:].set_linewidth(4)
plt.show()
