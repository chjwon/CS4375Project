from sklearn.datasets import fetch_20newsgroups
from bert_score import score
import pandas as pd
import sys

def dataload(dataName, dataType = 'all', data_label='data'):
    if dataName == '20NS':
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        
        if dataType == 'train':
            result = newsgroups_train
        elif dataType == 'test':
            result = newsgroups_test
        else:
            result = newsgroups        
        
        if data_label == 'data':
            return result['data']
        elif data_label == 'label':
            return result['data'].target
        elif data_label == 'clusterName':
            return result['data'].target_names
        else:
            print("Error in data_label passing")
            sys.exit()
    
    elif dataName == 'AGNEWS':
        agnews_train = pd.read_csv('./AGNEWS_data/train.csv')
        agnews_test = pd.read_csv('./AGNEWS_data/test.csv')
        agnews_all = pd.concat([agnews_train, agnews_test])
        if dataType == 'train':
            result = agnews_train
        elif dataType == 'test':
            result = agnews_test
        else:
            result = agnews_all

        if data_label == 'data':
            return result['Description'].to_list()
        elif data_label == 'label':
            return result['Class Index'].to_numpy(dtype=int)
        elif data_label == 'clusterName':
            return result['Title'].to_list()
        else:
            print("Error in data_label passing")
            sys.exit()
        
    

def keywordLoad(dataName):
    keyWordList = []
    raw = open('./'+dataName+'_keyWordList.txt','r')
    while True:
        line = raw.readline()
        if not line: break
        keyWordList.append(line[:-4])
    return keyWordList

def number_topic(model):
    return (max(model.get_topic_info()['Topic'])+1)

def model_topic_name(model, topics):
    pred = []
    name = model.get_topic_info()['Name']
    for i in range(len(topics)):
        pred.append(name[topics[i]+1])

    return pred


def total_bert_score(model, topics, name_label):
    pred = model_topic_name(model,topics)
    result = get_score_bert(pred,name_label)
    return result[0]

def get_score_bert(cands,refs):
    (P, R, F), hashname = score(cands, refs, lang="en", return_hash=True)
    # print(
    #     f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
    # )
    return P.mean().item(), P