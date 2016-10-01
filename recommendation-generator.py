import productVectors as p_vector
import pymongo
from pymongo import MongoClient
from bson.dbref import DBRef
import servicesimilarity as service
 

#server = 'ds025792.mlab.com'
#port = 25792
#db_name = 'itarget'
#username = 'itarget'
#password = 'itarget'
client = MongoClient('mongodb://127.0.0.1:27017/itarget')

db = client['itarget']
users = db['users']
resturants = db['resturants']


#user-based similarity algorithm

data = users.find({})
data_all={}
for i in data:
	temp={}
	for j in i["userRatings"]:
		temp[j["rest_id"]]=j["resturantVote"]
	if temp:
		data_all[i["_id"]]= temp

final={}
for item in data_all:
	final[item]= p_vector.getRecommendations(data_all,item)

for i in final:
	users.update({"_id":i},{ "$set": { "recommended_resturants": final[i] }})




################ item-based similarity algorithm #################

mongo_data_rest = resturants.find({})
data_all_rest={}
for i in mongo_data_rest:
	temp={}
	for j in i["rates"]:
		temp[j["user_id"]]=j["resturantVote"]
	if temp:
		data_all_rest[i["_id"]]= temp

final_rest={}
for item in data_all_rest:
	final_rest[item]= p_vector.getSimilarities(item,data_all_rest)

for i in final_rest:
	resturants.update({"_id":i},{ "$set": { "similarResturants": final_rest[i]
	}})


#####################sentiment analysis###################################
#from sklearn.externals import joblib
#from sframe import SFrame
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression

#_vocabulary = joblib.load('yelp_vocabulary_.pkl')
#vectorizer = CountVectorizer(vocabulary=_vocabulary)
##sample_test_matrix = vectorizer.transform(['ammazing wow wow'])
#clf = joblib.load('yelp_model.pkl')

#from string import punctuation

#def remove_punctuation(text):
#        return text.translate(punctuation)

#def get_sentiment(review_txt):
#	sample_test_matrix = vectorizer.transform([review_txt])
#	return str(clf.predict(sample_test_matrix)[0])


#mongo_data_rest = resturants.find({})
#data_all_rest = {}
#for resturant in mongo_data_rest:
#	rev_object = {}
#	reviews = []
#	for review in resturant["reviews"]:
#		rev_object = {
#			"user_id":review["user_id"],
#			"ReviewTxt":review["ReviewTxt"],
#			"review_date":review["review_date"],
#			"vote":review["vote"],
#			"sentiment":get_sentiment(remove_punctuation(review["ReviewTxt"]))
#			  }
#		reviews.append(rev_object)
#	data_all_rest[resturant["_id"]] = reviews

#	for i in data_all_rest:
#		resturants.update({"_id":i},{ "$set": { "reviews": data_all_rest[i]}})