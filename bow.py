from preprocessing import SUB_CORPUS, SIGN_CORPUS, PLANS_CORPUS, LOG_CORPUS, preprocess_text, SIGN_LABELS, PLANS_LABELS, SUB_LABELS, LOG_LABELS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

training_docs = SIGN_CORPUS + PLANS_CORPUS + SUB_CORPUS + LOG_CORPUS
training_labels = ['sign_up'] * len(SIGN_CORPUS) + ['plans'] * len(PLANS_CORPUS) + ['sub_accounts'] * len(SUB_CORPUS) + ['log_in'] * len(LOG_CORPUS)
training_labels_two = SIGN_LABELS + PLANS_LABELS + SUB_LABELS + LOG_LABELS
training_labels_combined = []

for x, y in zip(training_labels, training_labels_two):
    combined_result = x + " & " + y
    training_labels_combined.append(combined_result)


user_message = "I have a new device, how do I log in?"

bow_vectorizer = CountVectorizer()
training_vectors = bow_vectorizer.fit_transform(training_docs)
test_vectors = bow_vectorizer.transform(preprocess_text(user_message))

training_classifier = MultinomialNB()
training_classifier.fit(training_vectors, training_labels_combined)
predictions = training_classifier.predict(test_vectors)


def prediction(user_message):
    test_vectors = bow_vectorizer.transform(preprocess_text(user_message))
    predictions = training_classifier.predict(test_vectors)
    topic = predictions[0].split(' & ')[0]
    action = predictions[0].split(' & ')[1]
    return predictions, topic, action

user_message = 'What is the best plan for me?'


