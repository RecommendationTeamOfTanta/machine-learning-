import productVectors as p_vector
import pymongo
from pymongo import MongoClient
from bson.dbref import DBRef
 

#server = 'ds025792.mlab.com'
#port = 25792
db_name = 'itarget'
#username = 'itarget'
#password = 'itarget'

client = MongoClient('mongodb://itarget:itarget@ds025792.mlab.com:25792/itarget')
#client = MongoClient('mongodb://localhost:27017/itargetLocal')

db = client['itarget']
users = db['users']
resturants = db['resturants']

#insertion of resturants data
for k,v in p_vector.transformed_data_Set.items():
	rates = []
	for user_name,rate in v.items():
		user_rate_id = {}
		user_id = users.find_one({"name.first":user_name})["_id"]
		user_rate_id["user_id"] = user_id
		user_rate_id["resturantVote"] = rate
		rates.append(user_rate_id)

	#r = {
	#	'name':k,
	#	'rates':rates
	#	}
	#resturants.insert(r)

	resturants.update_one({ 'name': k },
			{ '$set': { 'rates': rates } }, upsert=True )

#insertion of users data
for k,v in p_vector.all_data_set.items():
	userRatings = []
	for rest_name,rate in v.items():
		rest_rate_id = {}
		rest_id = resturants.find_one({"name":rest_name})["_id"]
		rest_rate_id["rest_id"] = rest_id
		rest_rate_id["resturantVote"] = rate
		userRatings.append(rest_rate_id)

	#u = {
	#	'name':{'first':k}
	#	#'userRatings':userRatings
	#	}
	#users.insert(u)

	users.update_one({ 'name':{'first':k} },
			{ "$set": { 'userRatings': userRatings } },upsert=True)

