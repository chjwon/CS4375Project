from bertopic import BERTopic
import numpy as np

def bertopic_fit(emb, keywordList, nr_topics):
    topic_model = BERTopic(calculate_probabilities=False, 
                           verbose = True,embedding_model=emb, 
                           nr_topics = nr_topics)
    topics, probs = topic_model.fit_transform(keywordList)
    return topic_model, topics, probs

def model_load(dataName, emb, seed):
    topic_model = BERTopic.load("result/keybert_bertopic_{}_{}_{}".format(dataName, emb, seed))
    return topic_model

def topic_load(dataName, emb, seed):
    topics = np.load('result/keybert_bertopic_{}_{}_{}.npy'.format(dataName, emb, seed))
    return topics