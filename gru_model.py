import torch
import torch.nn as nn

device = 'cuda'

class LSTM_GRU(nn.Module):
    def __init__(self, word_vec_dim=25, LSTM_dim=128, GRU_dim=256,classes=2):
        """
        :param word_vec_dim: The dimension of word vector
        :param classes: The number of classes
        """
        super(LSTM_GRU, self).__init__()
        self.LSTM_dim = LSTM_dim
        self.GRU_dim = GRU_dim
        self.sentence_LSTM = nn.LSTM(input_size=word_vec_dim, hidden_size=LSTM_dim)
        self.tree_GRU = nn.GRU(input_size=LSTM_dim, hidden_size=GRU_dim)
        self.FC = nn.Linear(GRU_dim, classes)
        self.solfmax = nn.Softmax(dim=-1)

    def forward(self, input_tree):

        output_list = []
        LSTM_hidden = (torch.zeros(size=(1, 1, self.LSTM_dim)),
                       torch.zeros(size=(1, 1, self.LSTM_dim))) # hidden_dim: (num_layers*num_direction,  batch, hidden_size)
        for sentence in input_tree:
            # sentence.shape: [seq_length, input_size]
            sentence_input = torch.tensor(sentence)
            sentence_input = sentence_input.unsqueeze(dim=1)
            output, LSTM_hidden = self.sentence_LSTM(sentence_input, LSTM_hidden) # input_dim: (seq_length, batch, input_size)
            output_list.append(output)
        tree_represent = torch.cat(output_list, dim=0)

        GRU_hidden = torch.zeros(size=(1,1, self.GRU_dim))
        GRU_output, GRU_hidden = self.tree_GRU(tree_represent, GRU_hidden)

        FC_output = self.FC(GRU_output)

        res =  self.solfmax(FC_output)
        return res.squeeze(dim=0).squeeze(dim=0)




# The input format of LSTM_GRU:
'''
A list contains many trees
l = [tree1, tree2, ... treen]

Each tree contains many sentences:
l = [[sentence1, sentence2, ... sentencen], tree2, ...]

Each sentence contains several word vectors of same dimension

sentence = [vector1, vector2, ... vectorn] 

'''