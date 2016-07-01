


from math import sqrt
from data import critics
import operator

#############################################################
#eqlidian distance
##############################################################
def Sim_Euqlidean(prefs,person1,person2):
    sum_of_squares = 0
    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    # if they have no ratings in common, return 0
    if len(si) == 0: return 0

    # Add up the squares of all the differences

    #sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item],2)
    #for item in prefs[person1] if item in prefs[person2]])

    for item in prefs[person1]:
        if item in prefs[person2]:
            sum_of_squares += sum([pow(prefs[person1][item] - prefs[person2][item],2)])

    return 1 / (1 + sum_of_squares)



#############################################################
#Pearson correlatio
#############################################################
def Sim_Pearson(prefs,person1,person2):
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2] : si[item] = 1
    n = len(si)
    if n == 0:return 0

    #the sum of preferences
    sum1 = sum([prefs[person1][it] for it in si])
    sum2 = sum([prefs[person2][it] for it in si])

    #The sum of squares
    sum1Sq = sum([pow(prefs[person1][it],2) for it in si])
    sum2Sq = sum([pow(prefs[person2][it],2) for it in si])

    #the sum of products
    pSum = sum([prefs[person1][it] * prefs[person2][it] for it in si])

    # Calculate Pearson score
    # =(pSum-(sum1*sum2/n))/sqrt([sum1Sq-(pow(sum1,2)/n)]*[sum2Sq-(pow(sum2,2)/n)])
    score = (pSum - (sum1 * sum2 / n)) / sqrt((sum1Sq - pow(sum1,2) / n) * (sum2Sq - pow(sum2,2) / n))

    if score == 0:return 0
    return score

#############################################################
#Cosine Similarity
#############################################################
def Sim_Cosine(prefs,person1,person2):
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2] : si[item] = 1
    n = len(si)
    if n == 0:return 0

    #sum of products
    pSum = sum([prefs[person2][it] * prefs[person2][it] for it in si])

    sqrt1 = sqrt(sum([pow(prefs[person2][it],2) for it in si]))
    sqrt2 = sqrt(sum([pow(prefs[person2][it],2) for it in si]))

    pSqrt = sqrt1 * sqrt2
    if pSqrt == 0 : return 0

    score = pSum / pSqrt

    return score



#################################################################
# Ranking Method
#################################################################
def topMatches(prefs,person,n=5,similarity=Sim_Pearson):
    scores = [(similarity(prefs,person,other),other) for other in prefs if other != person]

    scores.sort()
    scores.reverse()
    return scores[0:n]


#################################################################
# Gets recommendations for a person by using a weighted average
#################################################################
def getRecommendations(prefs,person,similarity=Sim_Pearson):
    totals = {}
    SimSums = {}
    for other in prefs:
        # don't compare me to myself
        if other == person:continue
        sim = similarity(prefs,person,other)
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



###############################################################
#transforming function
#############################################################
def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person  
            result[item][person]=prefs[person][item]
    return result

movies = transformPrefs(critics)

#for item in getRecommendations(movies,"You, Me and Dupree",similarity=Sim_Pearson):
#    print(item)
#print('\n')
#for item in getRecommendations(movies,"You, Me and Dupree",similarity=Sim_Euqlidean):
#    print(item)
#print('\n')
#for item in getRecommendations(movies,"You, Me and Dupree",similarity=Sim_Cosine):
#    print(item)

#for item in topMatches(movies,'Lady in the Water'):
#    print(item)

#print("\n\n")

#for item in getRecommendations(critics,"Sleem",similarity=Sim_Pearson):
#    print(item)

#print("\n\n")
#print(transformPrefs(critics))
#print("\n\n")
#print(critics)

#print(Sim_Cosine(critics,"Atef","Hassan"))

#print(Sim_Pearson(critics,"Atef","Hassan"))

#print(Sim_Pearson(critics,"Atef","Hassan"))
print(getRecommendations(critics,"Hassan"))
print(topMatches(critics,"Hassan",4,Sim_Pearson));