import gensim
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils.datahelper import *

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics,
                                           id2word=dictionary, iterations=1000, workers=7, passes=5,
                                           minimum_probability=1e-8,
                                           minimum_phi_value=1e-8
                                           )
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

def lda_train(rawtext, num_topics, if_train=False):
    dictionary = gensim.corpora.Dictionary(rawtext)
    corpus = [dictionary.doc2bow(doc) for doc in rawtext]

    if if_train:
        lda = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics,
                                           id2word=dictionary, iterations=1000,
                                           workers=7, passes=10, minimum_probability=1e-8,
                                           minimum_phi_value=1e-8)
        dictionary.save('./lda_stuff/lda_model_T300.dictionary')
        lda.save('./lda_stuff/lda_model_T300.model')

    else:
        lda = gensim.models.LdaMulticore.load("./lda_stuff/lda_model_T300.model")


    topics = lda.state.get_lambda()
    topics_norm = topics / topics.sum(axis=1)[:, None]
    topics_norm = topics_norm.T

    word_topics = {}
    doc_topics = []


    for idx in range(len(corpus)):

        top_topics = lda.get_document_topics(corpus[idx])
        doc_topic = [top_topics[i][1] for i in range(num_topics)]
        doc_topics.append(doc_topic)

    with open('./word_topic_300_index.dict', 'wt') as file:
        count = 0
        for id in range(len(dictionary)):
            count += 1
            word_topics[lda.id2word[id]] = np.mean(topics_norm[id] * np.array(doc_topics), axis=0)
            out = [i for i in word_topics[lda.id2word[id]]]
            print(lda.id2word[id], out, sep='\t', file=file)

            if count % 500 == 0:
                print(count)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-train", default=False, type=bool, required=False, help="train the lda model if true")
    parser.add_argument("-H", default=True, type=bool, required=False, help="if true data would be 3D for hierarchical model")
    parser.add_argument("-T", default=300, type=int, required=False,
                        help="number of topics for training lda model")
    args = parser.parse_args()

    dataset = Dataset()
    dataset.data_reader(hierachical_data=args.H)
    raw_text = dataset.rawtext
    lda_train(raw_text, num_topics=args.T, if_train=args.train)