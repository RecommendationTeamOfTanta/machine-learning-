from math import sqrt
from data import critics



from data import critics as data

def CosinAveraged(person1,person2):
    bast = 0
    sqr1 = 0
    sqr2 = 0
    cond = False
    for k1 in person1:
        for k2 in person2:
            if(k1 < k2):
                break

            if(k1 == k2):
                p1 = person1[k1]
                p2 = person2[k2]
                bast+=p1 * p2
                sqr1+=p1 * p1
                sqr2+=p2 * p2
                cond = True
                break
    if (cond == False):
        return 0
    result = bast / ((sqrt(sqr1)) * (sqrt(sqr2)))
    return result

atef = {'1':3,'2':5,'10':1}
eissa = {'4':3,'3':2,'4':4}

data = {'atef' : {'1':3,'2':5,'10':1},'eissa' : {'2':3,'10':2,'11':4},'hassan' : {'4':3,'3':2,'4':4},
'magdy' : {'2':1,'10':2,'11':4},'sleem' : {'2':3,'10':2,'11':4},'engy' : {'1':3,'2':5,'10':1}}

result = CosinAveraged(atef,eissa)
r = ((5 * 3) + (1 * 2)) / ((sqrt(26)) * sqrt(13))



def getSimilarities(person,data):
    sim_score = {}
    sim_resut = 0
    

    for person2 in data:
        if(person != person2):
            sim_resut = CosinAveraged(data[person],data[person2])
            #sim_score.setdefault(person2,0)
            sim_score[person2] = sim_resut
    return sim_score


eissa = getSimilarities("Atef",critics)
for v in eissa.items():
     print(v)
