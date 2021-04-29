import torch, torch.nn as nn
import gru_model
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
device = 'cpu'

word_vector_dim = 25
LSTM_dim = 128
GRU_dim = 32
classes = 2
epochs = 100
lr = 0.005


dirname = 'data/'


def load_obj(name):
    with open(dirname + name + '.pkl', 'rb') as f:
        return pickle.load(f)


model = gru_model.LSTM_GRU(word_vec_dim=word_vector_dim,
                           LSTM_dim=LSTM_dim,
                           GRU_dim=GRU_dim,
                           classes=classes).to(device)
print('loaded_model, putting into ', device)

train_data = load_obj('train_set')
dev_data = load_obj('dev_set')
test_data = load_obj('test_set')
train_label = load_obj('train_label')
dev_label = load_obj('dev_label')
model_optimizer = torch.optim.SGD(
    # params=filter(lambda p: p.requires_grad, model.parameters()),
    params=model.parameters(),
    momentum=0.9,
    lr=lr
)

loss_func = nn.MSELoss(reduction='sum')

# train_data = [tree1, tree1, tree1]
# train_label = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
# TODO handle the hidden 问题出在hidden上面，添加了hidden以后就不行了

LSTM_hidden, GRU_hidden = model.init_hidden()

for iteration in range(epochs):
    train_loss = []
    for idx in tqdm(range(len(train_data))):
        if idx == 3126:
            continue    # The 3126th dataset is not complete
        #hidden = (tuple([each.data for each in hidden[0]]), [each.data for each in hidden[1]])
        LSTM_hidden = tuple([e.data for e in LSTM_hidden])
        GRU_hidden = GRU_hidden.data                                #
        pred, LSTM_hidden, GRU_hidden = model(train_data[idx], LSTM_hidden, GRU_hidden)
        # print(pred.shape)
        target = torch.FloatTensor(train_label[idx]).to(device)
        loss = loss_func(pred, target)
        train_loss.append(loss.data.cpu().numpy())
        model_optimizer.zero_grad()
        loss.backward(retain_graph=True)

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)      #梯度裁剪，暂且不用
        model_optimizer.step()
    print('epoch={}: loss={:.6f}s'.format(iteration, np.mean(train_loss)))
    with torch.no_grad():
        pred_list = []
        for idx in range(len(dev_data)):
            pred, _, _ = model(dev_data[idx], LSTM_hidden, GRU_hidden)
            pred = pred.to('cpu').numpy()
            if pred[0] > pred[1]:
                new_pred = 1
            else:
                new_pred = 0
            pred_list.append(new_pred)
        #print(dev_label[0], pred_list[0])
        print()
        acc = accuracy_score(dev_label, pred_list)
        f1 = f1_score(dev_label, pred_list, average='macro')
        p = precision_score(dev_label, pred_list, average='macro',zero_division=True)
        r = recall_score(dev_label, pred_list, average='macro')
        print('acc:', acc, ', F1:',f1, ' ,Precision:', p, ' ,Recall:', r)
    torch.save(model.state_dict(), 'check_points/model_'+str(iteration)+'.pt')

torch.save(model.state_dict(), 'check_points/final_model.pt')

'''

tree1 = [[[1.0 ,1.0 ,1.0 ,1.0 ,1.0, 1.0 ,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.00,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.0,1.0,1.0]]]


model = LSTM_GRU()
res = model(tree1)
print(res)


'''
