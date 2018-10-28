# COMP6714 Project
# DO NOT MODIFY THIS FILE!!!
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from todo import new_LSTMCell, get_char_sequence
import numpy as np



class sequence_labeling(nn.Module):

    def __init__(self, config, pretrain_word_embeddings, pretrain_char_embedding):
        super(sequence_labeling, self).__init__()

        self.config = config

        # employ the modified LSTM cell if the flag is True
        if self.config.use_modified_LSTMCell:
            torch.nn._functions.rnn.LSTMCell = new_LSTMCell

        self.word_embeds = nn.Embedding(self.config.nwords, self.config.word_embedding_dim)
        self.word_embeds.weight = nn.Parameter(torch.from_numpy(pretrain_word_embeddings).float())

        # below variants may be used for char embedding
        self.char_embeds = nn.Embedding(self.config.nchars, self.config.char_embedding_dim)
        self.char_embeds.weight = nn.Parameter(torch.from_numpy(pretrain_char_embedding).float())
        char_lstm_input_dim = self.config.char_embedding_dim
        self.char_lstm = nn.LSTM(char_lstm_input_dim, self.config.char_lstm_output_dim, 1, bidirectional=True)

        # employ char embedding if the flag is True
        if self.config.use_char_embedding:
            lstm_input_dim = self.config.word_embedding_dim + self.config.char_lstm_output_dim * 2#dim变150
        else:
            lstm_input_dim = self.config.word_embedding_dim
        self.lstm = nn.LSTM(lstm_input_dim, self.config.hidden_dim, 1, bidirectional=True)

        self.lstm2tag = nn.Linear(self.config.hidden_dim * 2, self.config.ntags)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.non_recurrent_dropout = nn.Dropout(self.config.dropout)

    def sort_input(self, seq_len):
        seq_lengths, perm_idx = seq_len.sort(0, descending=True)
        return perm_idx, seq_lengths

    def _rnn(self, batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists):
        input_word_embeds = self.word_embeds(batch_word_index_lists)
#        print('input_word_embeds: ',input_word_embeds.size())
#        print('batch_word_index_lists: ',batch_word_index_lists)
#        print('input: ',input_word_embeds)# 单一组[每句话[每个词[小数们]]].   
                                          # 每组多少句话[每句话长度[每个词多少小数[]]]
#        x=self.char_embeds(batch_char_index_matrices)
#        print('x:                         ',x)
#        x=self.forward(batch_word_index_lists, batch_sentence_len_list, batch_word_mask, batch_char_index_matrices, batch_word_len_lists, batch_char_mask, batch_tag_index_list)
#        print(x)
#        print('self.char_embeds.weight: ',self.char_embeds.weight)
#        x=np.reshape(x,-1)
#        print('x2:                         ',x)
        # employ char embedding if the flag is True
        if self.config.use_char_embedding:                 #
            output_char_sequence = get_char_sequence(self, batch_char_index_matrices, batch_word_len_lists)#每组句数不变，每句长度不变，小数变。
            input_embeds = self.non_recurrent_dropout(torch.cat([input_word_embeds, output_char_sequence], dim=-1))
        else:
            input_embeds = self.non_recurrent_dropout(input_word_embeds)
#            print('batch_word_len_lists: ',batch_word_len_lists)
#        print('batch_char_index_matrices: ',len(x[0][0][0]))
            
            
#        print('char_lstm_output: ',char_lstm_output[0][0])
#        print('temp_char_lstm_output: ',temp_char_lstm_output[0][0])       
            
#        print('temp_char_lstm_output: ',temp_char_lstm_output.size())
        
        
        
        
        
        perm_idx, sorted_batch_sentence_len_list = self.sort_input(batch_sentence_len_list)
#        a,b = self.sort_input(batch_word_len_lists)
        
        sorted_input_embeds = input_embeds[perm_idx]
#        sorted_x=x
#        print('perm_idx: ',perm_idx)
        _, desorted_indices = torch.sort(perm_idx, descending=False)
#        _,c = torch.sort(a,descending=False)
#        print('sorted_x: ',sorted_x)
#        print('sorted_input_embeds: ',sorted_input_embeds)
        output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_sentence_len_list.data.tolist(), batch_first=True)
#        y=pack_padded_sequence(sorted_x ,lengths=b.data.tolist(),batch_first=True )

        output_sequence, state = self.lstm(output_sequence)
#        y ,b= self.char_lstm(x[0])
        
        output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
#        y,_ = pad_packed_sequence(y,batch_first=True)
        
        output_sequence = output_sequence[desorted_indices]
#        y = y[c]
        
        output_sequence = self.non_recurrent_dropout(output_sequence)
#        y=self.non_recurrent_dropout(y)
        
#        print('output_sequence: ',output_sequence)
#        print('y:             : ',len(y[0][0]))
        logits = self.lstm2tag(output_sequence)
#        print('logits: ',logits)
        return logits

    def forward(self, batch_word_index_lists, batch_sentence_len_list, batch_word_mask, batch_char_index_matrices, batch_word_len_lists, batch_char_mask, batch_tag_index_list):
        logits = self._rnn(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices,
                           batch_word_len_lists)
        batch_tag_index_list = batch_tag_index_list.view(-1)
        batch_word_mask = batch_word_mask.view(-1)
        logits = logits.view(-1, self.config.ntags)
        train_loss = self.loss_func(logits, batch_tag_index_list) * batch_word_mask
        return train_loss.mean()

    def decode(self, batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists, batch_char_mask):
        
        logits = self._rnn(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices,
                           batch_word_len_lists)
        _, pred = torch.max(logits, dim=2)
        return pred