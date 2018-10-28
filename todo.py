#import torch
from config import config
from torch.nn import functional as F
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
_config = config()
# 'I-TAR',
#golden_list=[['B-TAR', 'I-TAR', 'B-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
#predict_list=[['B-TAR', 'I-TAR', 'I-TAR', 'O', 'O', 'O'], ['B-TAR', 'O', 'B-HYP', 'I-HYP']]

def evaluate(golden_list, predict_list):
    tp=tag1=tag2=0
#    print('golden_list',len(golden_list))
#    print('predict_list',len(predict_list))
    for i in range(len(golden_list)):
        list1=golden_list[i]#['B-TAR', 'I-TAR', 'O', 'B-HYP']
        list2=predict_list[i]#['B-TAR', 'O', 'O', 'O']
        n=0
        for j in range(len(list1)):
            if list1[j][0]=='B':
                tag1+=1
            if list2[j][0]=='B':
                tag2+=1
        while n<len(list1):
            if list1[n]!='O':
                if n==len(list1)-1:
                    if list2[n]==list1[n]:
                        tp+=1
                    break
                x=n
                b=0
                while list1[n]!='O' :
                    if list1[n][0]=='B' and b==1:
                        break
                    if list1[n][0]=='B':
                        b+=1
                    n+=1
                    if n==len(list1):
                        break
                ltemp=list1[x:n]
                if n<len(list1):
                    if list2[x:n]==list1[x:n] and list2[n][0]!='I':
                        tp+=1
                else:
                    if list2[x:n]==list1[x:n]:
                        tp+=1
            else:
                n+=1
    if tp ==tag1 == tag2 == 0:
        F1 = 1
    elif tp == 0:
        F1 = 0
    else:
        p = tp/tag2
        r = tp/tag1
        F1 = 2*(p*r)/(r+p)
    return F1


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    
    forgetgate = torch.sigmoid(forgetgate)
    ingate = 1-forgetgate
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    input_char_embeds = model.char_embeds(batch_char_index_matrices) # char embedding
    
    reshape_input_char_embeds = input_char_embeds.view(len(batch_word_len_lists)*len(batch_word_len_lists[0]),len(batch_char_index_matrices[0][0]),len(input_char_embeds[0][0][0]))# 给char_embeds的输出降维（4->3）
#    print(batch_word_len_lists.size())
    batch_word_len_lists=batch_word_len_lists.view(len(batch_word_len_lists)*len(batch_word_len_lists[0]))# 给word长度表降维
#    print('batch_word_len_lists',sum(batch_word_len_lists.tolist()))
    
    perm_idx, sorted_batch_word_len_list = model.sort_input(batch_word_len_lists)# 排序 储存排序方法
#    print('sorted_batch_word_len_list',sorted_batch_word_len_list)
    
    char_embeds = reshape_input_char_embeds[perm_idx]
    
    _, desorted_indices= torch.sort(perm_idx, descending=False)
    
    char_embeds = pack_padded_sequence(char_embeds, lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
    
    
#        print('reshape_input_char_embeds: ',reshape_input_char_embeds)
    char_embeds,(state,a) = model.char_lstm(char_embeds)
    size=list(state.size())
#    print('state:',size)
    output=state.view(list(state.size())[1],100)
#    print('state',state)
    i=0
    state=state.tolist()
    while i < size[1]:
        a=torch.Tensor(state[0][i])
        b=torch.Tensor(state[1][i])
        output[i]=torch.cat([a,b],dim=-1)
        i+=1

    output = output[desorted_indices]
#    print('outputsize:',output.size())
    output=output.view(len(batch_char_index_matrices),len(batch_char_index_matrices[0]),100)

    
    
    return output

#p=evaluate(golden_list, predict_list)
#print(p)