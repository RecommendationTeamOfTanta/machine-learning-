from math import sqrt
from math import pow
from data import critics as data
import csv
import random
import operator

def euc(person1,person2):

	euc_sqr = 0
	cond = False                         # to avoid such x/0 
	for k1 in person1:                  # k is the key value of the dict (id of restaurant)
		for k2 in person2:
			#if(k1 < k2):
			#    break

			if(k1 == k2):
				p1 = person1[k1]
				p2 = person2[k2]
				
				euc_sqr+=((p1-p2)*(p1-p2))                #euclidean algorithm
				cond =True
				
	if (cond==False):
		return 0
  
	euc_result=sqrt(euc_sqr)                                      #euclidean algorithm
  
	return euc_result


def pearson(person1,person2):
	

	pea_sub_p1 = 0                                    #pearson algorithm Variable   
	pea_sub_p2 = 0
	pea_sub_sq_p1 = 0
	pea_sub_sq_p2 = 0
	pea_sub_p1p2 = 0
	pea_result=0
	n=0
	cond = False                         # to avoid such x/0 
	for k1 in person1:                  # k is the key value of the dict (id of restaurant)
		for k2 in person2:
			if(k1 < k2):
				break

			if(k1 == k2):
				p1 = person1[k1]
				p2 = person2[k2]
							
				pea_sub_p1+=p1                                     #pearson algorithm    
				pea_sub_p2+=p2
				pea_sub_sq_p1+=(p1*p1)
				pea_sub_sq_p2+=(p2*p2)
				pea_sub_p1p2+=p1*p2
				n+=1
				cond =True
				break
	if (cond==False):
		return 0
	pea_result=(((n)*(pea_sub_p1p2))-((pea_sub_p1)*(pea_sub_p2)))/( sqrt(((n)*(pea_sub_sq_p1)-((pea_sub_p1)*(pea_sub_p1)))*((n)*(pea_sub_sq_p2)-((pea_sub_p1)*(pea_sub_p1)))))  #pearson algorithm result 
	#pea_result=((n*pea_sub_p1p2)-(pea_sub_p1*pea_sub_p2))/( sqrt((n*pea_sub_sq_p1-pow(pea_sub_p1,2))(n*pea_sub_sq_p2-pow(pea_sub_p2,2))) ) 
	return pea_result

def CosinAveraged(person1,person2):    # person1 , person2 is dictionary have (id of Restaurant) and (rating)
	cos_bast = 0
	cos_sqr1 = 0
	cos_sqr2 = 0
	#person11 = sorted(person1.items(), key=operator.itemgetter(0))
	#person22 = sorted(person2.items(), key=operator.itemgetter(0))
	cond = False                         # to avoid such x/0 
	for k1 in person1:                  # k is the key value of the dict (id of restaurant)
		for k2 in person2:
			#if(k1[0] < k2[0]):
			#    break

			if(k1 == k2):
				p1 =person1[k1]
				p2 =person2[k2]
				cos_bast+=p1 * p2                          #cos algorithm
				cos_sqr1+=p1 * p1
				cos_sqr2+=p2 * p2
				
				cond =True
				break
	if (cond==False):
		return 0
	cos_result = cos_bast / ((sqrt(cos_sqr1)) * (sqrt(cos_sqr2))) ##cos algorithm result
	return cos_result

def sim_distance(prefs,person1,person2):
# Get the list of shared_items
	si={}
	for item in prefs[person1]:
		if item in prefs[person2]:
			si[item]=1
			
	# if they have no ratings in common, return 0
	if len(si)==0: return 0
	
	# Add up the squares of all the differences
	sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
	for item in prefs[person1] if item in prefs[person2]])
	
	return 1/(1+sum_of_squares)



# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
	# Get the list of mutually rated items
	si={}
	for item in prefs[p1]:
		if item in prefs[p2]: si[item]=1
			
	# Find the number of elements
	n=len(si)
	
	# if they are no ratings in common, return 0
	if n==0: return 0
	
	# Add up all the preferences
	sum1=sum([prefs[p1][it] for it in si])
	sum2=sum([prefs[p2][it] for it in si])
	
	# Sum up the squares
	sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
	sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
	
	# Sum up the products
	pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
	
	# Calculate Pearson score
	num=pSum-(sum1*sum2/n)
	den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
	if den==0: return 0
	r=num/den
	return r



def getSimilarities(person,data):    # person is string and data dictionary      output is dict id of user and it similarity score
	sim_score = {}
	sim_resut = 0
	
	for person2 in data:
		if(person != person2):
			sim_resut = CosinAveraged(data[person],data[person2])
			sim_score[person2] = sim_resut
	return sim_score


def recommend(person_sim,person,data): 
	rec_res_total = {}
	rec_res_simSum={}
	rec_res={}
	for id_user in person_sim:   #give me one from the user with similarity
		sim=person_sim[id_user]  #save the similarity score in sim
		for id_res in data[id_user]: # give me the user rating dic key  ex (wavlez : 5)
			
			if id_res in rec_res_total.keys():
				rec_res_total[id_res]+=sim*data[id_user][id_res]
			else:
				rec_res_total[id_res]=sim*data[id_user][id_res]        
			if id_res in rec_res_simSum.keys():
				rec_res_simSum[id_res]+=sim
			else:
				 rec_res_simSum[id_res]=sim
	for key in rec_res_total:
		rec_res[key]=rec_res_total[key]/rec_res_simSum[key] 

	sorted_x =list(reversed(sorted(rec_res.items(), key=operator.itemgetter(1))))
	return(sorted_x)


def getRecommendations(prefs,person,similarity=CosinAveraged):
	totals = {}
	SimSums = {}
	for other in prefs:
		# don't compare me to myself
		if other == person:continue
		sim = similarity(prefs[person],prefs[other])
		# ignore scores of zero or lower
		if sim <= 0:continue
		for item in prefs[other]:
			if item not in prefs[person] or prefs[person][item] == 0:
				# Similarity * Score
				totals.setdefault(item,0)
				totals[item]+=prefs[other][item] * sim
				# Sum of similarities
				SimSums.setdefault(item,0)
				SimSums[item]+=sim
	#the ranking list
	rankings = [(total / SimSums[item],item) for item,total in totals.items()]
	rankings.sort()
	rankings.reverse()
	return rankings


def loadDataset(filename,all_data_set={},list_of_tuple=[]):
	
	with open(filename)  as csvfile:
		lines = csv.reader(csvfile)
		#to separate header of data
		headers = next(lines)[21:42]
		#prepare the traning set to be dictionary of lists
		dataset = list(lines)
		for row in dataset:
			all_data_set[row[0]]={}
			for h,v in zip(headers,row[21:42]):
				if v=="":
					continue
				all_data_set[row[0]][h]=float(v)
		

def transform_data(prefs):
	result={}
	for person in prefs:
		for item in prefs[person]:
			result.setdefault(item,{})
			# Flip item and person  
			result[item][person]=prefs[person][item]
	return result

def transformPrefs(prefs):
	result={}
	for person in prefs:
		for item in prefs[person]:
			result.setdefault(item,{})
			
			# Flip item and person
			result[item][person]=prefs[person][item]
	return result



def topMatches(prefs,person,n=5,similarity=sim_pearson):
	scores=[(similarity(prefs,person,other),other)
	for other in prefs if other!=person]
	
	# Sort the list so the highest scores appear at the top
	scores.sort()
	scores.reverse()
	return scores[0:n]

def calculateSimilarItems(prefs,n=10):
	# Create a dictionary of items showing which other items they
	# are most similar to.
	result={}
	
	# Invert the preference matrix to be item-centric
	itemPrefs=transformPrefs(prefs)
	
	c=0
	for item in itemPrefs:
		# Status updates for large datasets
		#c+=1
		#if c%100==0: print(c,len(itemPrefs))
		# Find the most similar items to this one
		scores=topMatches(itemPrefs,item,n=n,similarity=sim_pearson)
		result[item]=scores
	return result


def getRecommendedItems(prefs,itemMatch,user):
	userRatings=prefs[user]
	scores={}
	totalSim={}
	
	# Loop over items rated by this user
	for (item,rating) in userRatings.items():
		# Loop over items similar to this one
		for (similarity,item2) in itemMatch[item]:
			# Ignore if this user has already rated this item
			if item2 in userRatings: continue
			# Weighted sum of rating times similarity
			scores.setdefault(item2,0)
			scores[item2]+=similarity*rating
			# Sum of all the similarities
			totalSim.setdefault(item2,0)		
			totalSim[item2]+=similarity
			
	# Divide each total score by total weighting to get an average
	rankings=[(score/totalSim[item],item) for item,score in scores.items( )]
	# Return the rankings from highest to lowest
	rankings.sort()
	rankings.reverse()
	return rankings



#######################################
#TESTING
#######################################



all_data_set = {}#ourdata										
loadDataset("iTARGET.csv",all_data_set)
transformed_data_Set = transform_data(all_data_set)#transformed

def get_recommendations(user_id):
	#similarities = getSimilarities(user_id,all_data_set)
	#recommendations = recommend(similarities,user_id,all_data_set)
	recommendations =getRecommendations(all_data_set,user_id,similarity=euc)
	return recommendations

def get_rec_item_based(rest_name):
	sim_item_dataset=calculateSimilarItems(all_data_set)
	return sim_item_dataset[rest_name]

sim_item_dataset=calculateSimilarItems(all_data_set)
eissss=getRecommendations(all_data_set,"a44b92ffc230e5467234433ac3803960",similarity=euc)
#print(getRecommendedItems(all_data_set,sim_item_dataset,"a44b92ffc230e5467234433ac3803960"))

#sim = getSimilarities("Atef",data)
#eissaa = recommend(sim,'atef',data)

#test our new data
#similarities = getSimilarities("a44b92ffc230e5467234433ac3803960",all_data_set)
#recommendations = recommend(similarities,'a44b92ffc230e5467234433ac3803960',all_data_set)

#print(eissaa)
#print(recommendations)
#print("\n recommendations for user a44b92ffc230e5467234433ac3803960 : \n")
#for(k,v) in recommendations:
#	print(k+": "+str(v))
#for i in eissaa:
#    print("");
#    print(i,eissaa[i]);