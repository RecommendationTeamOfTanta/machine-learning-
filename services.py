from flask import Flask,jsonify,request,Response
import productVectors
import json

from sklearn.externals import joblib
from sframe import SFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

_vocabulary = joblib.load('yelp_vocabulary_.pkl')
vectorizer = CountVectorizer(vocabulary=_vocabulary)
#sample_test_matrix = vectorizer.transform(['ammazing wow wow'])
clf = joblib.load('yelp_model.pkl')

from string import punctuation

def remove_punctuation(text):
        return text.translate(punctuation)

app = Flask(__name__)

@app.route('/itarget/sentiment',methods=['GET'])
def get_sentiment():
	review_txt =remove_punctuation(request.args.get('rev_txt'))



	sample_test_matrix = vectorizer.transform([review_txt])

	#return jsonify(productVectors.get_recommendations(user_id))
	return jsonify(str(clf.predict(sample_test_matrix)[0]))



@app.route('/itarget/getrecommendation',methods=['POST'])
def get_recommendations():
	user_id = request.args.get('user_id')
	#return jsonify(productVectors.get_recommendations(user_id))
	return json.dumps(productVectors.get_recommendations(user_id)),
200,{'Content-Type':'application/json'}


							


app.run()