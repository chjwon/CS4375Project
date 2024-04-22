from gensim.models.doc2vec import Word2Vec
from transformers import GPT2Model

import numpy as np
import argparse
import random
import sys

from utils import dataload, keywordLoad, number_topic, total_bert_score
from keybert_main import get_KeyBert_result, keywordSave
from bertopic_main import bertopic_fit, model_load, topic_load



seed = 1
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS4375 Project')
    parser.add_argument("--lifelong", type=bool, default=False, help="regenerate keyword?")
    parser.add_argument("-d", "--dataset", type=str, default='20NS', help="dataset 20NS or AGNEWS")
    parser.add_argument("--emb", type=str, default='word2vec', help="embedding option word2vec or GPT2")
    # parser.add_argument("--gpt", type=bool, default=False, help="using GPT3 or not")
    parser.add_argument("--evaluation", type=bool, default=True, help="option to do evaluation")
    
    args_main = parser.parse_args()
    print(args_main)

    dataName = args_main.dataset
    lifelong = args_main.lifelong


    data = dataload(dataName=dataName, dataType='train', data_label='data')
    print("train data length: ",len(data))
    if lifelong:
        keywordList = get_KeyBert_result()
        keywordSave(dataName=dataName, keywordList=keywordList)

    keywordList = keywordLoad(dataName=dataName)

    if argparse.emb == 'word2vec':
        emb_method = Word2Vec
    elif argparse.emb == 'GPT2':
        emb_method = GPT2Model
    else:
        print('emb are not ready')
        sys.exit()

    if dataName == '20NS':
        nr_topics = 20
    elif dataName == 'AGNEWS':
        nr_topics = 4
    else:
        print('dataset are not ready')
        sys.exit()

    topic_model, topics, probs = bertopic_fit(emb=emb_method, keywordList=keywordList, nr_topics=nr_topics)
    topic_model.save('result/keybert_bertopic_{}_{}_{}'.format(dataName, argparse.emb, str(seed)))
    np_topics = np.array(topics)
    np.save("result/keybert_bertopic_{}_{}_{}".format(dataName, argparse.emb, str(seed),np_topics))

    if argparse.evaluation:
        data = dataload(dataName=dataName, dataType='test', data_label='data')
        
        # predicted
        model = model_load(dataName=dataName, emb=emb_method, seed=seed)
        topics = topic_load(dataName=dataName, emb=emb_method, seed=seed)

        # label in dataset
        label = dataload(dataName=dataName, dataType='test', data_label='label')
        clusterName = dataload(dataName=dataName, dataType='test', data_label='clusterName')
        name_label = []
        for i in range(len(label)):
            name_label.append(clusterName[label[i]])

        # evaluation
        pred_number_topic = number_topic(model=model)
        print("Predicted number of topics: ",pred_number_topic)
        print("Real number of topics: ", nr_topics)

        bert_score = total_bert_score(model=model, topics=topics, name_label=name_label)
        print("Bert Score in Cluster Name is: ", bert_score)

        
        