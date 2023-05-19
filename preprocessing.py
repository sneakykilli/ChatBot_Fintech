import pandas as pd
import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('ChatBot - corpus  - FinTech.csv')
response_df = df[(df['Topic'] == 'plans') & (df['Action'] == 'What')]
response = response_df['Resolution'].to_string(index=False)
response = response.replace('\\n', '\n')

#<<<<<<<<<<<<<< START BLOCK 1 >>>>>>>>>>>>>>>>>>>>>>>
#Group Question/Keyword by Topic.
#Merge together into 1 list per Topic.
list_topics = list(set(df['Topic'].to_list()))
TOPICS = ['sign_up', 'plans', 'sub_accounts', 'log_in']

sign_df = df[df['Topic'] == TOPICS[0]]
SIGN_MERGED = sign_df['Question'].to_list()
sign_action = sign_df['Action'].to_list()
sign_container = []

plans_df = df[df['Topic'] == TOPICS[1]]
PLANS_MERGED = plans_df['Question'].to_list()
plan_action = plans_df['Action'].to_list()
plans_container = []

sub_df = df[df['Topic'] == TOPICS[2]]
SUB_MERGED = sub_df['Question'].to_list()
sub_action = sub_df['Action'].to_list()
sub_container = []

log_df = df[df['Topic'] == TOPICS[3]]
LOG_MERGED = log_df['Question'].to_list()
log_action = log_df['Action'].to_list()
log_container = []
#<<<<<<<<<<<<<< END BLOCK 1 >>>>>>>>>>>>>>>>>>>>>>>

#<<<<<<<<<<<<<< START BLOCK 2 >>>>>>>>>>>>>>>>>>>>>>>
#Create Function to lemmatize, remove stop words and lower text.
def preprocess_text(text):
    """Takes in Text as a string and returns text with stop words removed, lemmatized and lowered"""
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        preprocessed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        preprocessed_sentence = ' '.join(preprocessed_words)
        preprocessed_sentences.append(preprocessed_sentence)

    return preprocessed_sentences


#<<<<<<<<<<<<<< END BLOCK 2 >>>>>>>>>>>>>>>>>>>>>>>

##<<<<<<<<<<<<<< START BLOCK 2 >>>>>>>>>>>>>>>>>>>>>>>
#Feed Topic into def above, append and flatted to produce corpus lists to be used in BOW vectorization.
for element in SIGN_MERGED:
    sign_container.append(preprocess_text(element))

SIGN_CORPUS = list(itertools.chain(*sign_container))
SIGN_LABELS = []
for index, element in enumerate(sign_action):
    SIGN_LABELS += [element] * len(sign_container[index])
# -------------------------------------------------------
for element in PLANS_MERGED:
    plans_container.append(preprocess_text(element))

PLANS_CORPUS = list(itertools.chain(*plans_container))
PLANS_LABELS = []
for index, element in enumerate(plan_action):
    PLANS_LABELS += [element] * len(plans_container[index])
# ------------------------------------------------------
for element in SUB_MERGED:
    sub_container.append(preprocess_text(element))

SUB_CORPUS = list(itertools.chain(*sub_container))
SUB_LABELS = []
for index, element in enumerate(sub_action):
    SUB_LABELS += [element] * len(sub_container[index])
# -------------------------------------------------------
for element in LOG_MERGED:
    log_container.append(preprocess_text(element))

LOG_CORPUS = list(itertools.chain(*log_container))
LOG_LABELS = []
for index, element in enumerate(log_action):
    LOG_LABELS += [element] * len(log_container[index])
##<<<<<<<<<<<<<< END BLOCK 2 >>>>>>>>>>>>>>>>>>>>>>>



