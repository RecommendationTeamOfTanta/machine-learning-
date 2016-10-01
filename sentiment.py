from sframe import SFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


products = SFrame('review.csv')

from string import punctuation

def remove_punctuation(text):
        return text.translate(punctuation)

products['review_clean'] = products['text'].apply(remove_punctuation)

products = products[products['stars'] != 3]

products['sentiment'] = products['stars'].apply(lambda r: +1 if r>3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])
print(test_matrix[0])


model = LogisticRegression()
model.fit(train_matrix, train_data['sentiment'])

sample_test_matrix = vectorizer.transform(['ammazing wow wow'])
print(sample_test_matrix)

model.decision_function(sample_test_matrix)

from sframe import SArray
def my_predictions(model, test_matrix):
    return SArray([+1 if s >= 0 else -1 for s in model.decision_function(test_matrix)])

print (my_predictions(model, sample_test_matrix))
print (SArray(model.predict(sample_test_matrix)))

#import pickle
#pickle.dumps(model)


from sklearn.externals import joblib
joblib.dump(model, 'yelp_model.pkl') 
joblib.dump(vectorizer.vocabulary_, 'yelp_vocabulary_.pkl')
#joblib.dump(vectorizer, 'vectroizer.pkl')


print (sum(model.coef_[0] >= 0))
print (float(sum(model.coef_[0] >= 0))/len(model.coef_[0]))
