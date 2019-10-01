from lda_han import LDA_HAN
from utils.normalize import normalize
import numpy as np
import argparse

SAVED_MODEL_DIR = 'checkpoints'
SAVED_MODEL_FILENAME = 'lda_HAN_best.h5'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-train", default="yes", type=str, required=False, help="train model if true")
    parser.add_argument("-txt", default="this is a test", type=str, required=False, help="test activation maps")
    parser.add_argument("-train_lda", default=False, type=bool, required=True, help="train LDA")
    args = parser.parse_args()

    if args.train=="yes":
        print("start to training")
        model = LDA_HAN()
        model.train(train_lda=args.train_lda)

    else:
        print("start to predicting")

        model = LDA_HAN()
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)

        txt = args.txt

        activation_maps = model.activation_maps(txt)
        print(activation_maps)

