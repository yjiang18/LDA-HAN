from lda_han import LDA_HAN
from utils.normalize import normalize
import numpy as np
import argparse

SAVED_MODEL_DIR = 'checkpoints'
SAVED_MODEL_FILENAME = 'lda_HAN_best.h5'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-train_lda", default=False, type=bool, required=True, help="train LDA")
    args = parser.parse_args()

    print("start to training")
    model = LDA_HAN()
    model.train(train_lda=args.train_lda)


