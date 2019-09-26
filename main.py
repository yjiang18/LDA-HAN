from lda_han import LDA_HAN
from utils.normalize import normalize
import numpy as np

SAVED_MODEL_DIR = 'checkpoints'
SAVED_MODEL_FILENAME = 'lda_HAN_best.h5'

if __name__ == '__main__':

    model = LDA_HAN()
    # model.train()
    model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)

    txt = "Google banned my channel with 80,000 people on it so join my Newsletter so they ca n't stop us !  A young man named Gio Rios just proved the FBI is lying scum about the Las Vegas shooting !  Everything we 've been told by the FBI and Fake News on TV is now officially a fraud !  Listen to his riveting testimony as he talks about the multiple shooters he saw and the lies the FBI told about it all !  Tell Trump to start firing people until the truth comes out !  This is outrageous !  We are worse than a banana Republic !  The FBI must be investigated for fraud and coverup !  We can not allow the FBI and the Fake News to get away with this any longer .  Spread this patriot 's testimony to every person you know .  Tell them the FBI are liars and our news is fake !  Tweet to Trump @potus and @realdonaldtrump and tell him God told us in his Word to tell the truth !  Tell him we all know the FBI is lying over Las Vegas !  Tell him to do a press conference and stop the lies !  Send him this video !  We are not going to take the lies anymore !   And Ye Shall Know the Truth , and the Truth Shall Make You Free  John 8:32  How To Easily Kill All Indoor Odor , Mold , And Bacteria ‚Äî Without Lifting A Finger"

    activation_maps = model.activation_maps(txt)
    print(activation_maps)


    norm_txt = normalize(txt)
    pred = model.predict(txt)
    print(pred)
    suma = np.sum(pred)
    print(suma)

