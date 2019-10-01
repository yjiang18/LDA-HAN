import numpy as np

def load_embedding(word_index, topic_word_dict):
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

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    lda_embedding_matrix = np.zeros((len(word_index) + 1, 425))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

        lda_vector = topic_word_dict.get(word)
        if lda_vector is not None:
            lda_embedding_matrix[i] = lda_vector

    return embedding_matrix, lda_embedding_matrix