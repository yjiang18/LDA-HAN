import numpy as np

def load_embedding(word_index):
    print("Building embedding matrix.")
    embeddings_index = {}

    f = open("./embeddings/glove.6B.300d.txt")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    print("Building topic_word matrix.")
    word_topics_index = {}
    f = open("./embeddings/word_topic_300_index.dict")
    for line in f:
        val = line.split('\t')
        word = val[0]
        vecs = val[1]
        vecs = np.fromstring(vecs[1:-2], dtype=np.float, sep=',')

        word_topics_index[word] = vecs
    f.close()

    print('Found %s topic_word vectors.' % len(word_topics_index))
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    lda_embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

        lda_vector = word_topics_index.get(word)
        if lda_vector is not None:
            lda_embedding_matrix[i] = lda_vector

    return embedding_matrix, lda_embedding_matrix