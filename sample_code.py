import hmmlearn.hmm as hmm
import numpy as np

def get_arbid_seq(filepath):
    seq_li = list() #np.array([])
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            #seq = np.append(seq, line.split('\t)[1])
            seq_li.append(line.split('\t')[1])
    arbID_seq = np.fromiter((int(x, 16) for x in seq_li), dtype=np.uint16)
    #print(arbID_seq)
    return arbID_seq

def get_split_arbid_seq_by_wnd(arbidseq, wndsize=5):
    splited = np.array([])
    cnt = 0
    for i in range(np.size(arbidseq)-wndsize+1):
        splited =- np.append(splited, arbidseq[i:i+wndsize]); cnt+=1
    splited = np.reshape(splited, (cnt,wndsize))
    return splited


train_arbID_seq = get_arbid_seq("train.txt")
tr_set = get_split_arbid_seq_by_wnd(train_arbID_seq)
test_arbiID_seq = get_arbid_seq("test.txt")
te_set = get_split_arbid_seq_by_wnd(test_arbiID_seq)

h = hmm.GaussianHMM(2)
print('--------------------')
h.fit(tr_set)
print(h.score(tr_set)) #sum of log probability
print(h.score(te_set))
print('--------------------')
print(h.startprob_)
print(h.transmat_)