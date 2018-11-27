import math
import metapy
import sys
import time


import pandas as pd
import seaborn as sns
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def tokenization(doc):
    #Write a token stream that tokenizes with ICUTokenizer, 
    #lowercases, removes words with less than 2 and more than 5  characters
    #performs stemming and creates trigrams (name the final call to ana.analyze as "trigrams")
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.LengthFilter(tok, min=2, max=5)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok = metapy.analyzers.ListFilter(tok, "stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)

    ana = metapy.analyzers.NGramWordAnalyzer(1, tok)
    unigrams = ana.analyze(doc)

    #leave the rest of the code as is
    tok.set_content(doc.content())
    tokens, counts = [], []
    for token, count in unigrams.items():
        counts.append(count)
        tokens.append(token)
    return tokens

    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    metapy.log_to_stderr()

    cfg = sys.argv[1]

    inv_idx = metapy.index.make_inverted_index(cfg)
    fwd_idx = metapy.index.make_forward_index(cfg)
    
    print('Print number of fwd_idx labels...{}'.format(fwd_idx.num_labels()))
    
    dset = metapy.classify.MulticlassDataset(fwd_idx)
    print('Print number of docs...{} '.format(len(dset)))
    print('Print label instance...{} '.format(set([dset.label(instance) for instance in dset])))
    
    print('Run LDA...')
    model = metapy.topics.LDAParallelGibbs(docs=dset, num_topics=10, alpha=0.1, beta=0.1)
    model.run(num_iters=1000)
    model.save('lda-pgibbs-customerreview')
    
    print('Load the results into memory for inspection...')
    model = metapy.topics.TopicModel('lda-pgibbs-customerreview')
    
    print('Print topics...')
    for topic in range(0, model.num_topics()):
        print("Topic {}:".format(topic + 1))
        for tid, val in model.top_k(topic, 5, metapy.topics.BLTermScorer(model)):
            print("{}: {}".format(fwd_idx.term_text(tid), val))
        print("======\n")
        
    
    print('Start Topic Over Time...')
    data = []
    for doc in dset:
       proportions = model.topic_distribution(doc.id)
       data.append([dset.label(doc)] + [proportions.probability(i) for i in range(0, model.num_topics())])
       #print('Print topic probability...{}'.format(proportions.probability(i)))
       
    df = pd.DataFrame(data, columns=['label'] + ["Topic {}".format(i + 1) for i in range(0, model.num_topics())])
    print(df)

    print('Start swarm plot...')
    for i in range(0, model.num_topics()):
        print("Topic {}".format(i + 1))
        sns.swarmplot(x='label', y="Topic {}".format(i + 1), data=df)
        plt.show() 


    #print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
    
