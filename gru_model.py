import torch
import torch.nn as nn

device = 'cuda'


class LSTM_GRU(nn.Module):
    def __init__(self, word_vec_dim=25, LSTM_dim=128, GRU_dim=256, classes=2):
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

    def init_hidden(self):
        LSTM_hidden  = (torch.zeros(size=(1, 1, self.LSTM_dim)),
                       torch.zeros(size=(1, 1, self.LSTM_dim))) # hidden_dim: (num_layers*num_direction,  batch, hidden_size)
        GRU_hidden = torch.zeros(size=(1, 1, self.GRU_dim))
        return LSTM_hidden, GRU_hidden

    def forward(self, input_tree):
        output_list = []  # Each treee keeps a output_list to hold all sentence outputs.
        # Each time forward a new tree, so the two hidden state must be NEW
        LSTM_hidden, GRU_hidden = self.init_hidden()
        ############################### sentence LSTM block ####################################################
        LSTM_hidden = (LSTM_hidden[0].to(device), LSTM_hidden[1].to(device))
        for sentence in input_tree:
            # sentence.shape: [seq_length, input_size]
            sentence_input = torch.tensor(sentence)
            sentence_input = sentence_input.unsqueeze(dim=1).to(device)
            output, LSTM_hidden = self.sentence_LSTM(sentence_input,
                                                     LSTM_hidden)  # input_dim: (seq_length, batch, input_size)
            # print(LSTM_hidden[0].shape, LSTM_hidden[1].shape)
            output_list.append(output[-1])
        # Up to now, tree_represent's shape : [sentence_num, word_embedding_dim]

        ############################## tree GRU block ########################################################
        tree_represent = torch.cat(output_list, dim=0)
        tree_represent = tree_represent.unsqueeze(dim=1)  # [sentence_num, 1, word_embedding_dim]
        # print(tree_represent.shape)
        GRU_hidden = GRU_hidden.to(device)
        GRU_output, GRU_hidden = self.tree_GRU(tree_represent, GRU_hidden)
        # print(GRU_output[-1].shape)
        FC_output = self.FC(GRU_output[-1])  # [1, GRU_hidden_dim]

        res = self.solfmax(FC_output)
        return res.squeeze(dim=0)


# The input format of LSTM_GRU:
'''
A list contains many trees
l = [tree1, tree2, ... treen]

Each tree contains many sentences:
l = [[sentence1, sentence2, ... sentencen], tree2, ...]

Each sentence contains several word vectors of same dimension

sentence = [vector1, vector2, ... vectorn] 

'''


class GRU(nn.Module):
    def __init__(self, GRU_dim=256, classes=2):
        """
        :param word_vec_dim: The dimension of word vector
        :param classes: The number of classes
        """
        super(GRU, self).__init__()
        self.GRU_dim = GRU_dim
        self.tree_GRU = nn.GRU(input_size=256, hidden_size=GRU_dim)
        self.FC = nn.Linear(GRU_dim, classes)
        self.solfmax = nn.Softmax(dim=-1)

    def forward(self, tree_represent, hidden=None):
        # input : [3, 1, self.GRU_dim]
        GRU_hidden = hidden
        GRU_output, GRU_hidden = self.tree_GRU(tree_represent, GRU_hidden)
        FC_output = self.FC(GRU_output[-1])  # [1, GRU_hidden_dim]
        return FC_output.squeeze(dim=0), GRU_hidden


if __name__ == "__main__":
    input = torch.randn(3, 1, 256)
    target = torch.randn(2)
    hidden = torch.zeros(size=(1, 1, 256))
    model = GRU(GRU_dim=256,
                classes=2).to(device)
    model_optimizer = torch.optim.SGD(
        # params=filter(lambda p: p.requires_grad, model.parameters()),
        params=model.parameters(),
        momentum=0.9,
        lr=0.1
    )

    loss_func = nn.MSELoss(reduction='sum')
    for i in range(10):
        hidden = hidden.data
        pred, hidden = model(input, hidden)

        loss = loss_func(pred, target)
        loss.backward()
        model_optimizer.step()


