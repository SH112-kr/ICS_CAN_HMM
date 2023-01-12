import hmmlearn.hmm as hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def normal_txt_to_csv(txt): #정상 txt경로 및 저장할 csv 경로 
    file = pd.read_csv(txt, delimiter = '\t',header=None,names=['TimeStamp','Arbitration','DLC','DATA','CLASS1'])
    csv = txt.replace(".txt",".csv")
    file.to_csv(csv)
    return csv
def abnormal_txt_to_csv(txt): #비정상 txt경로 및 저장할 csv 경로 
    file = pd.read_csv(txt, delimiter = '\t',header=None,names=['TimeStamp','Arbitration','DLC','DATA','CLASS1','CLASS2'])
    csv = txt.replace(".txt",".csv")
    file.to_csv(csv)
    return csv


def get_arbid_seq(filepath,seq_data): #filepath -> csv format
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
    if seq_data == '1':
        datas=csv_data['Arbitration']
        seq_li=datas.values.tolist()
        del seq_li[0]
        seq_result = np.fromiter((int(x, 16) for x in seq_li), dtype=np.uint16) 
        return seq_result
    if seq_data == '2':
        datas=csv_data['DATA']
        seq_li=datas.values.tolist()
        del seq_li[0]
        seq_result = np.fromiter((int(z, 16) for x in seq_li for z in x.split(' ')), dtype=np.uint16)
        return seq_result

    return seq_result


def get_split_arbid_seq_by_wnd(arbidseq, wndsize):
    splited = np.array([])
    splited = np.array([arbidseq[i:i+wndsize] for i in range(np.size(arbidseq)-wndsize+1)], dtype=np.float64) #uint16 에러 발생
    return splited

    #csv로 가공한 데이터 경로 지정 
def dir_path(train,test,seq_data):
    train_arbID_seq = get_arbid_seq(train,seq_data) #normal
    test_arbiID_seq = get_arbid_seq(test,seq_data) #attack
    return train_arbID_seq, test_arbiID_seq





def fit_model(windowsize, h_state, train_arbID_seq, test_arbiID_seq):
    score_distance = 0
    tr_set = get_split_arbid_seq_by_wnd(train_arbID_seq,windowsize)
    te_set = get_split_arbid_seq_by_wnd(test_arbiID_seq,windowsize)
    h = hmm.GaussianHMM(h_state) #hiddenstate 개수
    if seq_data == '1':
        print('      -------------------')
        print("       Arbitration Result ")
        print("      -------------------")
    elif seq_data == '2':
        print('      -------------')
        print("        DATA Result")
        print('      -------------')
    print("\n")
    print('--------------------------------------------------------------------------')
    h.fit(tr_set)
    score_trset = h.score(tr_set)
    score_teset = h.score(te_set)
    score_distance = abs(score_trset - score_teset) #Traing Log와 Test log 차이 계산
    print("Training log : ", score_trset) #sum of log probability
    print("Test log     : ", score_teset)
    print("윈도우 사이즈",windowsize," 의Traing log 와 Test log의 거리 : ",score_distance )
    print("초기 상태 확률 : ", h.startprob_)
    print("전이 확률 행렬 : \n", h.transmat_)
    print('--------------------------------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help ='학습 트레이닝 데이터 셋 경로')
    parser.add_argument('-t', help ='테스트 데이터 셋 경로')
    parser.add_argument('-o', help ='시퀀스 데이터 종류(숫자만 입력) 1. Arbitration, 2. DATA ')
    parser.add_argument('-w', help = ':windowsize 크기')
    parser.add_argument('-s', help = ':hidden state 개수')
    args = parser.parse_args()
    
    train_dir = args.e
    if ".txt" in train_dir: #데이터셋이 csv가 아니라 txt일 경우 변환해주는 함수 실행
        train_dir = normal_txt_to_csv(train_dir)
    test_dir = args.t
    if ".txt" in test_dir: #데이터셋이 csv가 아니라 txt일 경우 변환해주는 함수 실행
        test_dir = abnormal_txt_to_csv(test_dir)
    seq_data = args.o #숫자 받으면 변경 코드 추가
    wnd_size = args.w
    hidden_state = args.s
    
    train_arbID_seq, test_arbiID_seq =dir_path(train_dir,test_dir,seq_data)
    fit_model(int(wnd_size), int(hidden_state), train_arbID_seq, test_arbiID_seq)
