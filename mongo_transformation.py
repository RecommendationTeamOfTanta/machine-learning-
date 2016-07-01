import productVectors as p_vector
import pymongo
from pymongo import MongoClient
from bson.dbref import DBRef
 

server = 'ds025792.mlab.com'
port = 25792
db_name = 'itarget'
username = 'itarget'
password = 'itarget'

client = MongoClient('mongodb://itarget:itarget@ds025792.mlab.com:25792/itarget')

db = client['itarget']
users = db['users']
resturants = db['resturants']

#insertion of resturants data
for k,v in p_vector.transformed_data_Set.items():
	rates = []
	for user_name,rate in v.items():
		user_rate_id = {}
		#user_id = users.find_one({"name":user_name})["_id"]
		user_rate_id["user_id"] = user_name
		user_rate_id["resturantVote"] = rate
		rates.append(user_rate_id)

	r = {
		'name':k,
		'rates':rates
		}
	resturants.insert(r)


#insertion of users data
for k,v in p_vector.all_data_set.items():
	userRatings = []
	for rest_name,rate in v.items():
		rest_rate_id = {}
		rest_id = resturants.find_one({"name":rest_name})["_id"]
		rest_rate_id["rest_id"] = rest_id
		rest_rate_id["resturantVote"] = rate
		userRatings.append(rest_rate_id)

	u = {
		'name':{'first':k},
		'userRatings':userRatings
		}
	users.insert(u)


def loadDataset(filename,all_data_set={},list_of_tuple=[]):
	
	with open(filename)  as csvfile:
		lines = csv.reader(csvfile)
		#to separate header of data
		headers = next(lines)[21:42]
		#prepare the traning set to be dictionary of lists
		dataset = list(lines)
		for row in dataset:
			all_data_set[row[0]] = {}
			for h,v in zip(headers,row[21:42]):
				if v == "":
					continue
				all_data_set[row[0]][h] = float(v)


