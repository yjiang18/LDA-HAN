from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
from keras.utils import CustomObjectScope
from keras import initializers
from keras.engine.base_layer import Layer

import pickle
from utils.datahelper import *
import string
from spacy.lang.en import English
from utils.load_embedding import load_embedding
from utils.normalize import normalize
from utils.lda_gen import lda_train


class Attention(Layer):
    def __init__(self, regularizer=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.context = self.add_weight(name='context',
                                       shape=(input_shape[-1], 1),
                                       initializer=initializers.RandomNormal(
                                           mean=0.0, stddev=0.05, seed=None),
                                       regularizer=self.regularizer,
                                       trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
        attention = attention_in / K.expand_dims(K.sum(attention_in, axis=-1), -1)

        # if mask is not None:
        #     # use only the inputs specified by the mask
        #     # import pdb; pdb.set_trace()
        #     attention = attention * K.cast(mask, 'float32')

        weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class LDA_HAN():
    def __init__(self):
        self.model = None
        self.MAX_SENTENCE_COUNT = 30
        self.MAX_SENTENCE_LENGTH = 40
        self.word_attention_model = None
        self.tokenizer = None
        self.output_dim = 1

    def _load_embeddings(self):
        self.word_embedding, self.lda_embedding = load_embedding(self.tokenizer.word_index, self.topic_word_dict)

    def _build_model(self):
        sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32')

        self.embedding_layer = Embedding(len(self.tokenizer.word_index) + 1,
                                         300,
                                         weights=[self.word_embedding],
                                         trainable=True,
                                         mask_zero=False,
                                         name='word_embedding')

        self.lda_embedding_layer = Embedding(len(self.tokenizer.word_index) + 1,
                                             425,
                                             weights=[self.lda_embedding],
                                             trainable=True,
                                             mask_zero=False,
                                             name='lda_embedding')

        embedded_word_seq = self.embedding_layer(sentence_in)  # (batch_size, seq_len, word_embed)
        embedded_topic_seq = self.lda_embedding_layer(sentence_in)  # (batch_size, seq_len, topic_embed)

        topic_aware_embedding = Concatenate()([embedded_word_seq, embedded_topic_seq])

        # Generate word-attention-weighted sentence scores

        word_encoder = Bidirectional(GRU(50, return_sequences=True))(topic_aware_embedding)

        dense_transform_w = Dense(
            128,
            activation='relu',
            name='dense_transform_w')(word_encoder)

        word_out = Attention(name='word_attention')(dense_transform_w)

        attention_weighted_sentence = Model(sentence_in, word_out)
        self.word_attention_model = attention_weighted_sentence
        attention_weighted_sentence.summary()

        # Generate sentence-attention-weighted document scores
        texts_in = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH,), dtype='int32')
        doc_topics_in = Input(shape=(425,), dtype='float32')

        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(texts_in)

        sentence_encoder = Bidirectional(
            GRU(50, return_sequences=True))(attention_weighted_sentences)

        dense_transform_s = Dense(
            128,
            activation='relu',
            name='dense_transform_s')(sentence_encoder)

        attention_weighted_text = Attention(name='sentence_attention')(dense_transform_s)

        final_concat = Concatenate()([attention_weighted_text, doc_topics_in])
        prediction = Dense(self.output_dim, activation='sigmoid')(final_concat)
        self.model = Model([texts_in, doc_topics_in], prediction)
        self.model.summary()

        self.model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['acc'])

    def _load_data(self, train_lda=False):
        dataset = Dataset()
        dataset.data_reader(hierachical_data=True)
        x, y, word_index, self.tokenizer = dataset.load_data(hierachical_data=True)
        with open("./checkpoints/lda_HAN_token.tokenizer", 'wb') as outfile:
            pickle.dump(self.tokenizer, outfile)
        outfile.close()
        raw_text = dataset.rawtext
        doc_topics, self.topic_word_dict = lda_train(raw_text, num_topics=425, if_train=train_lda)
        train_seq, train_doc_topics, y_train, \
        test_seq, test_doc_topics, y_test = dataset.train_val_test(x, doc_topics, y)

        return np.array(train_seq), np.array(train_doc_topics), np.array(y_train), \
               np.array(test_seq), np.array(test_doc_topics), np.array(y_test)

    def train(self, train_lda=False):
        train_seq, train_doc_topics, y_train, \
        test_seq, test_doc_topics, y_test = self._load_data(train_lda=train_lda)

        self._load_embeddings()
        self._build_model()
        self.model.fit(x=[train_seq, train_doc_topics], y=y_train, validation_split=.1,
                                 batch_size=32, epochs=15,
                                 callbacks=[ModelCheckpoint(filepath='./checkpoints/lda_HAN_best_1.h5',
                                          monitor='val_acc',
                                          verbose=1, save_best_only=True, mode='max'),
                                            CSVLogger('./history/lda_HAN_training_1.log')])
        loss, acc = self.model.evaluate(x=[test_seq, test_doc_topics], y=y_test, verbose=1, batch_size=32)

        print("Model test accuracy is : %s " % acc)

    def _create_reverse_word_index(self):
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}

    def load_weights(self, saved_model_dir, saved_model_filename):
        with CustomObjectScope({'Attention': Attention}):
            self.model = load_model(os.path.join(saved_model_dir, saved_model_filename))
            self.word_attention_model = self.model.get_layer('time_distributed_1').layer
            self.tokenizer = pickle.load(open("./checkpoints/lda_HAN_token.tokenizer", "rb"))
            self._create_reverse_word_index()

    def _encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH))
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(self.tokenizer.texts_to_sequences(text),
                                                  maxlen=self.MAX_SENTENCE_LENGTH))[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][-len(encoded_text):] = encoded_text

        return encoded_texts

    def _encode_input(self, x):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])
        return self._encode_texts(texts)

    def activation_maps(self, text, websafe=False):
        normalized_text = normalize(text)
        encoded_text = self._encode_input(text)[0]

        # get word activations
        hidden_word_encoding_out = Model(inputs=self.word_attention_model.input,
                                         outputs=self.word_attention_model.get_layer('dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer('word_attention').get_weights()[0]
        u_wattention = encoded_text * np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))
        if websafe:
            u_wattention = u_wattention.astype(float)

        # generate word, activation pairs
        nopad_encoded_text = encoded_text[-len(normalized_text):]
        nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
        reconstructed_texts = [[self.reverse_word_index[int(i)]
                                for i in sentence] for sentence in nopad_encoded_text]
        nopad_wattention = u_wattention[-len(normalized_text):]
        nopad_wattention = nopad_wattention / np.expand_dims(np.sum(nopad_wattention, -1), -1)
        nopad_wattention = np.array([attention_seq[-len(sentence):]
                                     for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])
        word_activation_maps = []
        for i, text in enumerate(reconstructed_texts):
            word_activation_maps.append(list(zip(text, nopad_wattention[i])))

        # get sentence activations
        hidden_sentence_encoding_out = Model(inputs=self.model.input,
                                             outputs=self.model.get_layer('dense_transform_s').output)
        hidden_sentence_encodings = np.squeeze(
            hidden_sentence_encoding_out.predict(np.expand_dims(encoded_text, 0)), 0)
        sentence_context = self.model.get_layer('sentence_attention').get_weights()[0]
        u_sattention = np.exp(np.squeeze(np.dot(hidden_sentence_encodings, sentence_context), -1))
        if websafe:
            u_sattention = u_sattention.astype(float)
        nopad_sattention = u_sattention[-len(normalized_text):]

        nopad_sattention = nopad_sattention / np.expand_dims(np.sum(nopad_sattention, -1), -1)

        activation_map = list(zip(word_activation_maps, nopad_sattention))

        return activation_map

    def predict(self, text):

        encoded_text = self._encode_input(text)
        return self.model.predict(encoded_text)