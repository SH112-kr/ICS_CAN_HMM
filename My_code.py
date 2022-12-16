import hmmlearn.hmm as hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_arbid_seq(filepath): #filepath -> csv format
    """
    0:'Data Number'
    1:'TimeStamp'
    2:'Arbitation'
    3:'DLC'
    4:'DATA'
    5:'CLASS1' -- Normal or Attack(No edu data)
    6:'CLASS2' -- Normal or Attack KindO(No edu data)
    """
    seq_li = list() #np.array([])
    csv_data = pd.read_csv(filepath)
    datas=csv_data['Arbitation']
    seq_li=datas.values.tolist()
    del seq_li[0]
    arbID_seq = np.fromiter((int(x, 16) for x in seq_li), dtype=np.uint16) 
    return arbID_seq

def get_split_arbid_seq_by_wnd(arbidseq, wndsize):
    splited = np.array([])
    splited = np.array([arbidseq[i:i+wndsize] for i in range(np.size(arbidseq)-wndsize+1)], dtype=np.float64) #uint16 에러 발생
    return splited


train_arbID_seq = get_arbid_seq("C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (normal only).csv") #normal
test_arbiID_seq = get_arbid_seq("C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (attack included).csv") #attack



windowsize = [5,50,100,200,300,400,500,600,700,800,900,1000]
x = list()
y = list()
z = list()
distance_gap = list()
previous_distance = 0
score_distance = 0
for i in windowsize:
    x.append(i)
    tr_set = get_split_arbid_seq_by_wnd(train_arbID_seq,i)
    te_set = get_split_arbid_seq_by_wnd(test_arbiID_seq,i)
    h = hmm.GaussianHMM(2) #hiddenstate 개수
    print('------------------------------------------------------------')
    h.fit(tr_set)
    score_trset = h.score(tr_set)
    score_teset = h.score(te_set)
    previous_distance = score_distance
    score_distance = abs(score_trset - score_teset) #Traing Log와 Test log 차이 계산
    distance_gap= score_distance - previous_distance
    print("Training log : ", score_trset) #sum of log probability
    print("Test log     : ", score_teset)
    print("윈도우 사이즈",i," 의Traing log 와 Test log의 거리 : ",score_distance )
    print("이전 거리와의 차이",distance_gap)
    y.append(score_distance)
    z.append(distance_gap)
    print("초기 상태 확률 : ", h.startprob_)
    print("전이 확률 행렬 : \n", h.transmat_)
    print('------------------------------------------------------------')
#print(x)
#print(y)
z[0] = 0
plt.subplot(1,2,1)
plt.plot(x,y)
plt.title("Traing - Tes")
plt.subplot(1,2,2)
plt.plot(x,z)
plt.title("Previous Distance")
plt.show()