# =============================================================================
import numpy as np 
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
# =============================================================================


# =============================================================================
class Agent:
    def __init__(self,ID):
        self.ID = ID
        self.gender = random.choice(["M","F"])
        if self.gender == "F":
            self.kaiji = np.random.normal(0.7,0.02)#女性の開示度の分布の平均値が男性より高い
            self.course = random.choice([1,2,3,4])#コースを選ばせる。確率が同じ
        else:
            self.course = "no"
            self.kaiji = np.random.normal(0.5,0.01)
        self.income = random.lognormvariate(2.6, 0.7)#対数正規分布
        if self.income <= 100:
            self.income = 100
        self.charm = np.random.randint(0,100)#個人魅力
        self.age = 0
        self.spire = np.random.rand()#意欲
        self.emotion = random.choice([1,0])#結婚に対する感情
        self.prob = 0#主観的な確率
        self.married = 0#結婚状態
        self.happiness = 0#幸福度
        self.net = []
        self.name_list = []
        self.m_list = []
        self.f_list = []
        friend_conut = np.random.choice(["A","B","C","D"],p=[0.1,0.5,0.3,0.1])
        if friend_conut == "A":
            self.friend_conut = 0
        elif friend_conut == "B":
            fri_c = np.random.choice([1,2,3,4,5,6])
            self.friend_conut = fri_c
        elif friend_conut == "C":
            fri_c = np.random.choice([7,8,9,10,11,12,13,14,15])
            self.friend_conut = fri_c
        elif friend_conut == "D":
            fri_c = np.random.choice([16,17,18,19,20])
            self.friend_conut = fri_c
        self.couple = []
        self.fri_man=[]
        self.fri_woman = []
        self.fri_married = []
        self.mark = 0
        self.data = []
        self.data.append(self.spire)
        self.gaibu = 0
        self.data2 = {}
        for i in range(100):
            self.data2[i] = []
        
        
    def happiness_change(self,x,y):
        i.happiness = 1 - math.sqrt(((i.income- x)**2 + (i.charm - y)**2)/20000)

# =============================================================================



# =============================================================================
N = 200#エージェント総数
step = 50
agent = []
for i in range(N):
    agent.append(Agent(i))
bunpu = []
bunpu_m = []
bunpu_w = []
for i in agent:
    bunpu.append(i.spire)
    if i.gender == "M":
        if i.married == 0:
            bunpu_m.append(i.spire)
    else:
        if i.married == 0:
            bunpu_w.append(i.spire)
# =============================================================================


# =============================================================================
G = nx.Graph()#自作ネットワーク
for i in range(N):
    G.add_node(i,gender=agent[i].gender,emotion=agent[i].emotion)
G2 = nx.gnp_random_graph(N,0.2)#ランダムグラフ
# =============================================================================


# =============================================================================
for i in range(N):
    G2.nodes[i]["属性"] = agent[i] 
# =============================================================================

# =============================================================================
#nネットワークを作るための準備
m_ID = []
f_ID = []
general_ID = []
for i in agent:
    general_ID.append(i.ID)
    if i.gender == "M":
        m_ID.append(i.ID)
    else:
        f_ID.append(i.ID)
for i in agent:
    i.name_list = copy.copy(general_ID)
    i.m_list = list(copy.copy(m_ID))
    i.f_list = list(copy.copy(f_ID))
for i in agent:
    if (i.ID in i.m_list):
        i.m_list.remove(i.ID)
    if (i.ID in i.f_list):
        i.f_list.remove(i.ID)
# =============================================================================
# =============================================================================
#初期既婚エージェント
kansen = 20
marry_man = list(copy.copy(m_ID))
marry_woman = list(copy.copy(f_ID))
for i in range(kansen):
    man = np.random.choice(marry_man)
    woman = np.random.choice(marry_woman)
    agent[man].married = 1
    agent[man].couple.append(woman)
    agent[woman].married = 1
    agent[woman].couple.append(man)
    marry_man.remove(man)
    marry_woman.remove(woman)
# =============================================================================

# =============================================================================
#ネットワークの形成
for i in agent:
    if i.friend_conut == 0:
        continue
    else:
        if len(i.net) == 0:
            for x in range(i.friend_conut):
                if i.gender == "M":
                    link = np.random.choice(["m_ID","f_ID"],p=[0.8,0.2])
                    if link == "m_ID":
                        friend = np.random.choice(i.m_list)
                        i.net.append(agent[friend].ID)
                        agent[friend].net.append(i.ID)
                    else:
                        friend = np.random.choice(i.f_list)
                        i.net.append(agent[friend].ID)
                        agent[friend].net.append(i.ID)
                else:
                    link = np.random.choice(["m_ID","f_ID"],p=[0.2,0.8])
                    if link == "f_ID":
                        friend = np.random.choice(i.f_list)
                        i.net.append(agent[friend].ID)
                        agent[friend].net.append(i.ID)
                    else:
                        friend = np.random.choice(i.m_list)
                        i.net.append(agent[friend].ID)
                        agent[friend].net.append(i.ID)
        elif len(i.net) != 0:
            if len(i.net) - i.friend_conut < 0:
                for i in range(len(i.net) - i.friend_conut):
                    if i.gender == "M":
                        link = np.random.choice(["m_ID","f_ID"],p=[0.8,0.2])
                        if link == "m_ID":
                            friend = np.random.choice(i.m_list)
                            i.net.append(agent[friend].ID)
                            agent[friend].net.append(i.ID)
                        else:
                            friend = np.random.choice(i.f_list)
                            i.net.append(agent[friend].ID)
                            agent[friend].net.append(i.ID)
                    else:
                        link = np.random.choice(["m_ID","f_ID"],p=[0.2,0.8])
                        if link == "f_ID":
                            friend = np.random.choice(i.f_list)
                            i.net.append(agent[friend].ID)
                            agent[friend].net.append(i.ID)
                        else:
                            friend = np.random.choice(i.m_list)
                            i.net.append(agent[friend].ID)
                            agent[friend].net.append(i.ID)
# =============================================================================

# =============================================================================
for i in agent:
    for x in i.net:
        G.add_edge(i,x)
nx.draw(G)
# =============================================================================


# =============================================================================
# =============================================================================



# =============================================================================
#初期既婚エージェントの幸福度計算
for i in agent:
    if i.married == 1:
        couple = i.couple[0]
        i. happiness_change(agent[couple].income,agent[couple].charm)
# =============================================================================


# =============================================================================
#シミュレーションのステップ
naibu_k = 0.2
gaibu_k = 0.5
prob_k = 0.5
spire_k = 2/3
for t in range(step):
    for i in agent:
        if i.married == 1:
            i.data.append(i.spire)#既婚であったら下のステップに入らない
        else:
            if i.gender == "M":#男性であったら下のステップに入る
                charm_list = []
                income_list = []
                if len(i.net) == 0:
                    i.spire = naibu_k*(1-(math.log((t+1),step)))**2+(spire_k*(i.spire))
                    i.data2[0].append(i.spire)
                    i.data.append(i.spire)
                else:
                    net_kibo = len(i.net)
                    for j in i.net:#まず自分のネットワークを確認する
                        charm_list.append(agent[j].charm)
                        income_list.append(agent[j].income)
                        if agent[j].married == 0:#相手が未婚エージェントなら次のステップに入る
                            if (agent[j].gender == "M"):#相手が男性であったら次のステップに入る
                                if agent[j].emotion != i.emotion:#結婚感情が違う場合のみ相互作用が起こると想定する
                                    i.fri_man.append(agent[j].ID)#違う感情を持つ男性と女性それぞれのリストを作る
                            else:
                                if agent[j].emotion != i.emotion:
                                    i.fri_woman.append(agent[j].ID)
                        elif agent[j].married == 1:
                            i.fri_married.append(agent[j].ID)
                    try:
                        i.prob = prob_k*(math.sqrt((i.income - np.mean(income_list))**2+(i.charm - np.mean(charm_list))**2)/math.sqrt((np.mean(income_list))**2+(np.mean(charm_list))**2))#主観的な確率を計算する
                    except ZeroDivisionError:
                        i.prob = prob_k *0.5
                    if len(i.fri_man) != 0:#まず自分と違うエージェントがあるかどうかを確認してから相互作用をする
                        if len(i.fri_man) == 1:
                            result = np.random.choice(["切る","切らない"],p=[0.5,0.5])
                            if result == "切る":
                                i.net.remove(agent[i.fri_man[0]].ID)
                                agent[i.fri_man[0]].net.remove(i.ID)
                            else:
                                change = np.random.rand()
                                result = np.random.choice(["自分","相手"],p=[0.5,0.5])
                                if result == "自分":
                                    i.mark = 1
                                else:
                                    agent[i.fri_man[0]].mark = 1
                        elif len(i.fri_man) > 1:
                            friend_act_m = np.random.choice(i.fri_man)
                            if i.ID not in agent[friend_act_m].net:
                                print("pass")
                                time.sleep(3)
                                continue
                            elif i.ID in agent[friend_act_m].net:
                                result = np.random.choice(["切る","切らない"],p=[0.5,0.5])
                                if result == "切る":
                                    i.net.remove(friend_act_m)
                                    i.fri_man.remove(agent[friend_act_m].ID)
                                    agent[friend_act_m].net.remove(i.ID)
                                    while len(i.fri_man) != 0:
                                            friend_act_m2 = np.random.choice(i.fri_man)
                                            result2 = np.random.choice(["切る","切らない"],p=[0.5,0.5])
                                            print("結果",result2)
                                            if result2 == "切る":
                                                i.net.remove(agent[friend_act_m2].ID)
                                                i.fri_man.remove(agent[friend_act_m2].ID)
                                                agent[friend_act_m2].net.remove(i.ID)
                                            elif result == "切らない":
                                                change = np.random.rand()
                                                result = np.random.choice(["自分","相手"],p=[change,1-change])
                                                if result == "自分":
                                                    i.mark = 1
                                                else:
                                                    agent[friend_act_m].mark = 1                
                            else:
                                change = np.random.rand()
                                result = np.random.choice(["自分","相手"],p=[change,1-change])
                                if result == "自分":
                                    i.mark = 1
                                else:
                                    agent[i.fri_man[0]].mark = 1
                        else:
                            i.mark = 0
                    if len(i.fri_woman) != 0:
                        if len(i.fri_woman) == 1:
                            if i.kaiji >= agent[i.fri_woman[0]].kaiji:
                                reslut = np.random.choice(["変える","変えない"],p=[0.2,0.8])
                                if result == "変える":
                                    agent[i.fri_man[0]].mark = 1
                                else:
                                    reslut = np.random.choice(["変える","変えない"],p=[0.2,0.8])
                                    if result == "変える":
                                        i.mark = 1
                        if len(i.fri_woman) > 1:
                            friend_act_w = np.random.choice(i.fri_woman)
                            bingo = np.random.rand()
                            result = np.random.choice(["変える","変えない"],p=[bingo,1-bingo])
                            if i.kaiji >= agent[i.fri_woman[0]].kaiji:
                                reslut = np.random.choice(["変える","変えない"],p=[0.2,0.8])
                                if result == "変える":
                                    agent[friend_act_w].mark = 1
                                else:
                                    i.mark = 1
                            else:
                                reslut = np.random.choice(["変える","変えない"],p=[0.2,0.8])
                                if result == "変える":
                                    i.mark = 1
                                else:
                                    agent[friend_act_w].mark = 1
                    elif len(i.fri_woman) == 0:
                        i.mark = 0
                    if len(i.fri_married) != 0:
                        if len(i.fri_married) == 1:
                            i.gaibu = gaibu_k*((1/(2)**(t+1))*(agent[i.fri_married[0]].happiness/0.5)**2)
                        if len(i.fri_married) > 1:
                            married_list = []
                            for j3 in i.fri_married:
                                married_list.append(agent[j3].happiness)
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*(np.mean(married_list)/0.5)**2)
                    if len(i.fri_married) == 0:
                        i.gaibu = 0
                    if i.mark == 1 and i.emotion == 0:
                        i.spire = naibu_k*(1-(math.log((t+1),step)))**2+(spire_k*(i.spire))+i.gaibu+i.prob
                        if i.spire < 0:
                            i.spire = 0
                        elif i.spire > 1:
                            i.spire = 1
                        elif np.isnan(i.spire) == True:
                            i.spire = 0
                            i.spire = np.mean(i.data)
                        i.data.append(i.spire)
                    elif i.mark == 1 and i.emotion == 1:
                        i.spire = (spire_k*(i.spire))+i.gaibu+i.prob-(naibu_k)*(1-(math.log((t+1),step)))**2
                        if i.spire < 0:
                            i.spire = 0
                        elif i.spire > 1:
                            i.spire = 1
                        elif np.isnan(i.spire) == True:
                            i.spire = 0
                            i.spire = np.mean(i.data)
                        i.data.append(i.spire)
                    elif i.mark == 0 and i.emotion == 0:
                        i.spire = (spire_k*(i.spire))+i.gaibu+i.prob-(naibu_k)*(1-(math.log((t+1),step)))**2
                        if i.spire < 0:
                            i.spire = 0
                        elif i.spire > 1:
                            i.spire = 1
                        elif np.isnan(i.spire) == True:
                            i.spire = 0
                            i.spire = np.mean(i.data)
                        i.data.append(i.spire)
                    elif i.mark == 0 and i.emotion == 1:
                        i.spire = naibu_k*(1-(math.log((t+1),step)))**2+(spire_k*(i.spire))+i.gaibu+i.prob
                        if i.spire < 0:
                            i.spire = 0
                        elif i.spire > 1:
                            i.spire = 1
                        elif np.isnan(i.spire) == True:
                            i.spire = 0
                            i.spire = np.mean(i.data)
                        i.data.append(i.spire)
                    i.data2[net_kibo].append(i.spire)
            if i.gender == "F":
                if i.net == 0:
                    i.spire = naibu_k*(1-(math.log((t+1),step)))**2+(spire_k*(i.spire))
                    i.data.append(i.spire)
                    i.data2[0].append(i.spire)
                else:
                    net_kibo = len(i.net)
                    charm_list = []
                    income_list = []
                    for j in i.net:
                        charm_list.append(agent[j].charm)
                        income_list.append(agent[j].income)
                        if agent[j].married == 0:
                            if (agent[j].gender == "M"):
                                if agent[j].emotion != i.emotion:
                                    i.fri_man.append(j)
                            else:
                                if agent[j].emotion != i.emotion:
                                    print()
                                    i.fri_woman.append(agent[j].ID)
                        elif agent[j].married == 1:
                            i.fri_married.append(agent[j].ID) 
                    try:
                        i.prob = prob_k*(math.sqrt((i.income - np.mean(income_list))**2+(i.charm - np.mean(charm_list))**2)/math.sqrt((np.mean(income_list))**2+(np.mean(charm_list))**2))#主観的な確率を計算する
                    except ZeroDivisionError:
                        i.prob = prob_k *0.5
                    if len(i.fri_woman) != 0:
                        if len(i.fri_woman) == 1:
                            change = np.random.rand()
                            result = np.random.choice(["相手","自分"],p=[change,1-change])
                            if result == "自分":
                                i.mark = 1
                            else:
                                agent[i.fri_woman[0]].mark = 1
                        elif len(i.fri_woman) > 1:
                            if len(i.fri_woman) == 1:
                                change = np.random.rand()
                                result = np.random.choice(["相手","自分"],p=[change,1-change])
                                if result == "自分":
                                    i.mark = 1
                                else:
                                    agent[i.fri_woman[0]].mark = 1
                            elif len(i.fri_woman) > 1:
                                same = 0
                                diff = 0
                                for i2 in i.fri_woman:
                                    if agent[i2].emotion == i.emotion:
                                        same = same + 1
                                    else:
                                        diff = diff + 1
                                if same == diff:
                                    change = np.random.rand()
                                    result = np.random.choice(["変える","変えない"],p=[change,1-change])
                                    if result == "変える":
                                        i.mark = 1
                                    elif result == "変えない":
                                        i.mark = 0
                                elif same > diff:
                                    i.mark = 0
                                elif diff > same:
                                    i.mark = 1                                
                    else:
                        i.mark = 0
                    if len(i.fri_married) != 0:
                        if len(i.fri_married) == 1:
                            if agent[i.fri_married[0]] == "M":
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*(agent[i.fri_married[0]].happiness/0.5)**(1/2))
                            elif agent[i.fri_married[0]] == "F" and agent[i.fri_married[0]].course == i.course:
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*(agent[i.fri_married[0]].happiness/0.5)**2)
                            elif agent[i.fri_married[0]] == "F" and agent[i.fri_married[0]].course != i.course:
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*(agent[i.fri_married[0]].happiness/0.5)**(1/2))
                        elif len(i.fri_married) > 1:
                            married_list_same = []
                            married_list_diff = []
                            married_diffgen = []
                            for j3 in i.fri_married:
                                if agent[j3].course == i.course:
                                    married_list_same.append(agent[j3].happiness)
                                elif agent[j3].course != i.course and agent[j3].course == "no":
                                    married_list_diff.append(agent[j3].happiness)                                
                            if len(married_list_diff) == 0 and len(married_list_same) != 0:    
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*(agent[i.fri_married[0]].happiness/0.5)**2)
                            elif len(married_list_same) == 0 and len(married_list_diff) != 0:
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*(agent[i.fri_married[0]].happiness/0.5)**(1/2))
                            elif len(married_list_same) != 0 and len(married_list_diff) != 0:
                                i.gaibu = gaibu_k*((1/(2)**(t+1))*((np.mean(married_list_diff)/0.5)**(1/2))+(np.mean(married_list_same)/0.5)**2)
                    else:
                        gaibu = 0
                if i.mark == 1 and i.emotion == 1:
                    i.spire = (spire_k*(i.spire))+ i.gaibu+ i.prob-(naibu_k)*(1-(math.log((t+1),step)))**2
                    if i.spire < 0:
                        i.spire = 0
                    elif i.spire > 1:
                        i.spire = 1
                    elif np.isnan(i.spire) == True:
                        i.spire = 0
                        i.spire = np.mean(i.data)
                    i.data.append(i.spire)
                elif i.mark == 1 and i.emotion == 0:
                    i.spire = naibu_k*(1-(math.log((t+1),step)))**2+(spire_k*(i.spire))+ i.gaibu+ i.prob
                    if i.spire < 0:
                        i.spire = 0
                    elif i.spire > 1:
                        i.spire = 1
                    elif np.isnan(i.spire) == True:
                        i.spire = 0
                        i.spire = np.mean(i.data)
                    i.data.append(i.spire)
                elif i.mark == 0 and i.emotion == 1:
                    i.spire = naibu_k*(1-(math.log((t+1),step)))**2+(spire_k*(i.spire))+ i.gaibu+ i.prob
                    if i.spire < 0:
                        i.spire = 0
                    elif i.spire > 1:
                        i.spire = 1
                    elif np.isnan(i.spire) == True:
                        i.spire = 0
                        i.spire = np.mean(i.data)
                    i.data.append(i.spire)
                elif i.mark == 0 and i.emotion == 0:
                    i.spire = (spire_k*(i.spire))+ i.gaibu+ i.prob-(naibu_k)*(1-(math.log((t+1),step)))**2
                    if i.spire < 0:
                        i.spire = 0
                    elif i.spire > 1:
                        i.spire = 1
                    elif np.isnan(i.spire) == True:
                        i.spire = 0
                        i.spire = np.mean(i.data)
                    i.data.append(i.spire)
                i.data2[net_kibo].append(i.spire)
            if i.mark == 1:
                i.emotion = abs(i.emotion -1 )
                i.mark = 0
            i.m_list.clear()
            i.f_list.clear()
            i.fri_man.clear()
            i.fri_woman.clear()
            i.fri_married.clear()
            i.m_list = list(copy.copy(m_ID))
            i.f_list = list(copy.copy(f_ID))
        man = np.random.choice(marry_man)
    woman = np.random.choice(marry_woman)
    agent[man].married = 1
    agent[man].couple.append(woman)
    agent[woman].married = 1
    agent[woman].couple.append(man)
    marry_man.remove(man)
    marry_woman.remove(woman)

# =============================================================================
# =============================================================================
#確認
bunpu2 = []
bunpu_m2 = []
bunpu_w2 = []
for i in agent:
    bunpu2.append(i.spire)
    if i.gender == "M":
        if i.married == 0:
            bunpu_m2.append(i.spire)
    else:
        if i.married == 0:
            bunpu_w2.append(i.spire)
# =============================================================================
#変化推移
x = np.arange(0, 201)

for i in m_ID:
    plt.plot(x,agent[i].data)

for i in f_ID:
    plt.plot(x,agent[i].data)

#友人ネットワーク規模と結婚意欲の散布図ー全体
all_net = {}
for i in range(100):
    all_net[i] = []
len(all_net[1]) 
for i in agent:
    for j in range(100):
        if len(i.data2[j]) == 0:
            continue
        else:
            all_net[j].append(np.mean(i.data2[j]))
all_data = list(all_net.items())
all_data_x = []
all_data_y = []
for i in range(100):
    for j in range(len(all_data[i][1])):
        all_data_x.append(all_data[i][0])
        all_data_y.append(all_data[i][1][j])
plt.xlim(-1,25)
plt.ylim(0,1)
plt.grid(True)
plt.scatter(all_data_x,all_data_y)



#男女別の散布図
man_net = {}
woman_net = {}
for i in range(100):
    man_net[i] = []
    woman_net[i] = []
for i in agent:
    if i.gender == "M":
        for j in range(100):
            if len(i.data2[j]) == 0:
                continue
            else:
                man_net[j].append(np.mean(i.data2[j]))
    else:
            for j in range(100):
                if len(i.data2[j]) == 0:
                    continue
                else:
                    woman_net[j].append(np.mean(i.data2[j]))


man_data = list(man_net.items())
woman_data = list(woman_net.items())
man_data_x = []
man_data_y = []
woman_data_x = []
woman_data_y = []

#man_data
for i in range(100):
    for j in range(len(man_data[i][1])):
        man_data_x.append(man_data[i][0])
        man_data_y.append(man_data[i][1][j])
plt.xlim(-1,25)
plt.ylim(0,1)
plt.grid(True)
plt.scatter(man_data_x,man_data_y)
#woman_data
for i in range(100):
    for j in range(len(woman_data[i][1])):
        woman_data_x.append(woman_data[i][0])
        woman_data_y.append(woman_data[i][1][j])
plt.xlim(-1,25)
plt.ylim(0,1)
plt.grid(True)
plt.scatter(woman_data_x,woman_data_y,color="orange")
 

h = []
for i in range(len(bunpu_m)):
    h.append(int(0))
plt.scatter(h,bunpu_m)

h2 =[]
for i in range(len(bunpu_m2)):
    h2.append(1)
plt.scatter(h2,bunpu_m2)

h3 = []
for i in range(len(bunpu_w)):
    h3.append(0)
plt.scatter(h3,bunpu_w)
h4 = []
for i in range(len(bunpu_w2)):
    h4.append(1)
plt.scatter(h4,bunpu_w2)