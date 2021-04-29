# LSTM_GRU


### 数据预处理  
gensim word2vec,每一个出现的词都分配了一个vector (下一步应该考虑只分配给一定词频以上的词)
词的预处理做了tokenization, lowercase, remove non-English words, lemma
(还应该进一步删除很奇怪的非英文词例如emoji,日语等)
