import torch, torch.nn as nn
import gru_model
device = 'cpu'

word_vector_dim = 25
LSTM_dim = 128
GRU_dim = 256
classes = 2
epochs = 100
lr = 0.005

import pickle
dirname = 'data/'
def load_obj(name ):
    with open( dirname + name + '.pkl', 'rb') as f:
        return pickle.load(f)

model = gru_model.LSTM_GRU(word_vec_dim=word_vector_dim,
                           LSTM_dim=LSTM_dim,
                           GRU_dim=GRU_dim,
                           classes=classes).to(device)
print('loaded_model, putting into ', device)

train_data = load_obj('train_set')
dev_data = load_obj('dev_set')
test_data = load_obj('test_set')

model_optimizer = torch.optim.SGD(
	# params=filter(lambda p: p.requires_grad, model.parameters()),
	params=model.parameters(),
	momentum=0.9,
	lr=lr
)

loss_func = nn.MSELoss(reduction='sum')

tree1 = [[[1.0 ,1.0 ,1.0 ,1.0 ,1.0, 1.0 ,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.00,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.0,1.0,1.0]]]
train_data = [tree1, tree1, tree1]
train_label = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]

for iter in range(epochs):
    for idx in range(len(train_data)):
        pred = model(train_data[idx])
        target = torch.FloatTensor(train_label[idx])
        loss = loss_func(pred, target)
        model_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model_optimizer.step()
    if epochs/5==0:
        with torch.no_grad:
            for idx in range(len(dev_data)):
                pred = model(dev_data[idx])
                





'''

tree1 = [[[1.0 ,1.0 ,1.0 ,1.0 ,1.0, 1.0 ,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.00,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.0,1.0,1.0]]]


model = LSTM_GRU()
res = model(tree1)
print(res)


'''