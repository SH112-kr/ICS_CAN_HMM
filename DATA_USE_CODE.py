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
    datas=csv_data['DATA'] #DATA 태그 읽어오기
    seq_li=datas.values.tolist()
    del seq_li[0] #태그 삭제
    print(seq_li)
    arbID_seq = np.fromiter((int(z, 16) for x in seq_li for z in x.split(' ')), dtype=np.uint16) #DATA 값 10진수로 변환
    return arbID_seq

def get_split_arbid_seq_by_wnd(arbidseq, wndsize):
    splited = np.array([])
    #print(np.size(arbidseq)-wndsize+1)
    #print(arbidseq[0:wndsize])
    #print(arbidseq[1:1+wndsize])

    splited = np.array([arbidseq[i:i+wndsize] for i in range(np.size(arbidseq)-wndsize+1)], dtype=np.uint16) #uint16 에러 발생
    
    #cnt = 0
    #for i in range(np.size(arbidseq)-wndsize+1): #for문 사용해서 배열에 append reshape하는것보단 array가 효율적일것이라 판단 -> 성능 향상
    #    splited = np.append(splited, arbidseq[i:i+wndsize]); cnt+=1
    #splited = np.reshape(splited, (cnt,wndsize))
    return splited


train_arbID_seq = get_arbid_seq("C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (normal only).csv") #normal
test_arbiID_seq = get_arbid_seq("C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (attack included).csv") #attack


x = list()
y = list()
z = list()
distance_gap = list()
previous_distance = 0
for i in range(1,1000):
    x.append(i)
    tr_set = get_split_arbid_seq_by_wnd(train_arbID_seq,i)
    te_set = get_split_arbid_seq_by_wnd(test_arbiID_seq,i)
    h = hmm.GaussianHMM(2) #hiddenstate 개수
    print('------------------------------------------------------------')
    h.fit(tr_set)
    score_trset = h.score(tr_set)
    score_teset = h.score(te_set)
    #previous_distance = score_distance
    score_distance = abs(score_trset - score_teset) #Traing Log와 Test log 차이 계산
    distance_gap= score_distance - previous_distance
    print("Training log : ", score_trset) #sum of log probability
    print("Test log     : ", score_teset)
    print("윈도우 사이즈",i," 의Traing log 와 Test log의 거리 : ",score_distance )
    print("이전 거리와의 차이",distance_gap)
    y.append(distance_gap)
    z.append(score_distance)
    print("초기 상태 확률 : ", h.startprob_)
    print("전이 확률 행렬 : \n", h.transmat_)
    print('------------------------------------------------------------')


y[0] = 0
print(y)
plt.plot(x,y,'.')
plt.show()