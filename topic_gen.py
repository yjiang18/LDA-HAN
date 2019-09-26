from utils.normalize import normalize
from utils.lda_gen import *
import gensim
import logging

txt = "Shutterstock  A Republican student association at San Diego State University is facing backlash for sending a letter demanding Muslim students condemn last week 's terror attacks in Barcelona .  The Republican Club at San Diego State University wrote to the Muslim Student Association , claiming that the campus community could not move forward creating ‚Äò an inclusive environment for all students ' until radical terrorism was disavowed  .  The letter said if no condemnation was forthcoming , the leaders of the student group should be forced to resign .  The letter , signed by SDSU College Republicans Chairman Brandon Jones , stated in part that  until radical Islamic terrorism is disavowed by the Muslim Student Organization at SDSU , we can not move forward in creating an inclusive environment for all students on campus .  It added the Muslim Student Association 's leadership should resign if they do not disavow Islamic terrorism .  The national Muslim Student Association expressed support for the San Diego State chapter for  their solidarity , strength and perseverance in the face of ignorance and hate .   The Young Democratic Socialists of SDSU responded by declaring :  We condemn the San Diego State College Republicans ' disgraceful statement towards the SDSU Muslim Student Association and the SDSU Muslim community .  Retract and apologize now . The Transfronterizo Alliance Student Organization , which describes itself as working to create an  inclusive campus environment for SDSU students who live a transborder lifestyle , joined the chorus ."
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)

def new_document(text):

    dictionary = gensim.corpora.Dictionary.load("./lda_stuff/lda_model_T300.dictionary")

    norm_text = normalize(text)

    norm_text = [word for sent in norm_text for word in sent.split()]
    new_corpus = [dictionary.doc2bow(norm_text)]

    lda = gensim.models.LdaMulticore.load("./lda_stuff/lda_model_T300.model")

    # lda.update(new_corpus)

    vec = lda[new_corpus[0]]

    all_vec = [vec[i][1] for i in range(300)]
    topics = max(all_vec)

    l = [i for i, j in enumerate(all_vec) if j == topics]

    for topic in l:
        print(lda.print_topic(topic))


new_document(txt)