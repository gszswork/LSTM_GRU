from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader

import jsonlines


import pickle
dirname = 'data/'


def save_obj(obj, name ):
    with open(dirname+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open( dirname + name + '.pkl', 'rb') as f:
        return pickle.load(f)

########################### load data from project-data path#####################################
path = 'project-data/'
train_data_path = 'train.data.jsonl'
dev_data_path = 'dev.data.jsonl'
test_data_path = 'test.data.jsonl'

train_data = []
with jsonlines.open(path + train_data_path) as reader:
    for obj in reader:
        train_data.append(obj)
print('length of traininig data:', len(train_data))

# load the development data as dev_data(used as test_data in RvNN)
dev_data = []
with jsonlines.open(path + dev_data_path) as reader1:
    for obj in reader1:
        dev_data.append(obj)
print('length of devolop data: ', len(dev_data))

test_data = []
with jsonlines.open(path + test_data_path) as reader2:
    for obj in reader2:
        test_data.append(obj)
print('length of test data: ', len(test_data))

import json

train_label_path = 'train.label.json'
dev_label_path = 'dev.label.json'
with open(path + train_label_path) as f1, open(path + dev_label_path) as f2:
    train_label = json.load(f1)
    dev_label = json.load(f2)



# Sort the three dataset in time_stamp order

# Wed Jan 07 14:04:29 +0000 2015
def time_stamp(dic):
    # given a tweet dict
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', "Aug", 'Sep', 'Oct', 'Nov', 'Dec']
    time = dic['created_at']
    month = time[4:7]
    day = time[8:10]
    h = time[11:13]
    m = time[14:16]
    s = time[17:19]
    return month_list.index(month)*86400*30 + int(day)*86400 + int(h)*3600 + int(m)*60 + int(s)*1


for idx in range(len(train_data)):
    train_data[idx] = sorted(train_data[idx], key=lambda tree_node: time_stamp(tree_node))

for idx in range(len(test_data)):
    test_data[idx] = sorted(test_data[idx], key=lambda tree_node: time_stamp(tree_node))

for idx in range(len(dev_data)):
    dev_data[idx] = sorted(dev_data[idx], key=lambda tree_node: time_stamp(tree_node))

tl = []
for idx in range(len(train_data)):
    Id = train_data[idx][0]['id_str']
    l = train_label[Id]
    if l == 'rumour':
        tl.append([1.0, 0.0])
    if l == 'non-rumour':
        tl.append([0.0, 1.0])

dl = []
for idx in range(len(dev_data)):
    Id = dev_data[idx][0]['id_str']
    l = dev_label[Id]
    if l == 'rumour':
        dl.append(1)
    if l == 'non-rumour':
        dl.append(0)

save_obj(tl, 'train_label')
save_obj(dl, 'dev_label')


# Get All texts and process them into Text and Dev
Train_txt = []
for tree in train_data:
    sub_text = []
    for elem in tree:
        sub_text.append(elem['text'])
    Train_txt.append(sub_text)

Dev_txt = []
for tree in dev_data:
    sub_text = []
    for elem in tree:
        sub_text.append(elem['text'])
    Dev_txt.append(sub_text)

Test_txt = []
for tree in test_data:
    sub_text = []
    for elem in tree:
        sub_text.append(elem['text'])
    Test_txt.append(sub_text)

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import copy, re  # We need regular expression to filter tokens.

tt = TweetTokenizer()
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = set(stopwords.words('english'))  # note: stopwords are all in lowercase
# import these modules
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

Text_list = [Train_txt, Dev_txt, Test_txt]


def preprocess_text(data, dev, test):
    '''
    1. Tokenize the tweets
    2. lowercase and remove non-English words and stopwords
    3. Lemma
    '''
    word_dict = {}  # Word dictionary
    res = []  # All processed texts
    sentences = []

    x = copy.deepcopy(data)
    processed_train = []
    for tree in x:
        sub_res = []
        for elem in tree:
            # 1. Tokenize the Tweets
            temp = tt.tokenize(elem)
            # 2. Trans all word into lowercase
            temp = [token.lower() for token in temp]
            # 3. Remove non-English words and remove stopwords
            temp = [token for token in temp if re.match(r'.*[a-z]+.*', token)
                    and token not in stopwords]

            # 4. Word Lemmatize
            # HTTP 和 @ 视作单独一类
            for idx in range(len(temp)):
                if temp[idx][:4] == 'http':
                    temp[idx] = 'http'
                if temp[idx][:1] == '@':
                    temp[idx] = '@'
                if temp[idx][:1] == '#':
                    temp[idx] = temp[idx][1:]
                temp[idx] = lemma.lemmatize(temp[idx])

                if temp[idx] not in word_dict:
                    word_dict[temp[idx]] = 1
                else:
                    word_dict[temp[idx]] += 1

            sentences.append(temp)
            sub_res.append(temp)
        processed_train.append(sub_res)

    x1 = copy.deepcopy(dev)
    processed_dev = []
    for tree in x1:
        sub_res = []
        for elem in tree:
            temp = tt.tokenize(elem)
            temp = [token.lower() for token in temp]
            temp = [token for token in temp if re.match(r'.*[a-z]+.*', token)
                    and token not in stopwords]
            for idx in range(len(temp)):
                if temp[idx][:4] == 'http':
                    temp[idx] = 'http'
                if temp[idx][:1] == '@':
                    temp[idx] = '@'
                if temp[idx][:1] == '#':
                    temp[idx] = temp[idx][1:]
                temp[idx] = lemma.lemmatize(temp[idx])
                if temp[idx] not in word_dict:
                    word_dict[temp[idx]] = 1
                else:
                    word_dict[temp[idx]] += 1
            sentences.append(temp)
            sub_res.append(temp)
        processed_dev.append(sub_res)

    x2 = copy.deepcopy(test)
    processed_test = []
    for tree in x2:
        sub_res = []
        for elem in tree:
            temp = tt.tokenize(elem)
            temp = [token.lower() for token in temp]
            temp = [token for token in temp if re.match(r'.*[a-z]+.*', token)
                    and token not in stopwords]
            for idx in range(len(temp)):
                if temp[idx][:4] == 'http':
                    temp[idx] = 'http'
                if temp[idx][:1] == '@':
                    temp[idx] = '@'
                if temp[idx][:1] == '#':
                    temp[idx] = temp[idx][1:]
                temp[idx] = lemma.lemmatize(temp[idx])
                if temp[idx] not in word_dict:
                    word_dict[temp[idx]] = 1
                else:
                    word_dict[temp[idx]] += 1
            sentences.append(temp)
            sub_res.append(temp)
        processed_test.append(sub_res)

    res = [processed_train, processed_dev, processed_test]
    return res, word_dict, sentences


new_Text, word_dict, sentences = preprocess_text(Train_txt, Dev_txt, Test_txt)
print('The corpus size: ', len(word_dict))
print(new_Text[0][0][0][0])

train_set, dev_set, test_set = new_Text[0], new_Text[1], new_Text[2]
model = Word2Vec(sentences=sentences, vector_size=25, min_count=1, workers=4)
for idx in range(len(train_set)):
    # tree: train_set[idx]
    for sen_idx in range(len(train_set[idx])):
        # sentence: train_set[idx][sen_idx]
        for word_idx in range(len(train_set[idx][sen_idx])):
            # word: train_set[idx][sen_idx][word_idx]
            # print(train_set[idx][sen_idx][word_idx])
            word_vec = model.wv[train_set[idx][sen_idx][word_idx]]
            train_set[idx][sen_idx][word_idx] = word_vec

for idx in range(len(dev_set)):
    # tree: dev_set[idx]
    for sen_idx in range(len(dev_set[idx])):
        # sentence: dev_set[idx][sen_idx]
        for word_idx in range(len(dev_set[idx][sen_idx])):
            # word: dev_set[idx][sen_idx][word_idx]
            # print(dev_set[idx][sen_idx][word_idx])
            word_vec = model.wv[dev_set[idx][sen_idx][word_idx]]
            dev_set[idx][sen_idx][word_idx] = word_vec

for idx in range(len(test_set)):
    # tree: test_set[idx]
    for sen_idx in range(len(test_set[idx])):
        # sentence: test_set[idx][sen_idx]
        for word_idx in range(len(test_set[idx][sen_idx])):
            # word: test_set[idx][sen_idx][word_idx]
            # print(test_set[idx][sen_idx][word_idx])
            word_vec = model.wv[test_set[idx][sen_idx][word_idx]]
            test_set[idx][sen_idx][word_idx] = word_vec

save_obj(train_set, 'train_set')
save_obj(dev_set,   'dev_set')
save_obj(test_set,  'test_set')
