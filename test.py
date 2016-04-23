import operator
import ast

data = {'atef' : {'1':3,'2':5,'10':1},'eissa' : {'2':3,'10':2,'11':4},'hassan' : {'4':3,'3':2,'4':4},
'magdy' : {'2':1,'10':2,'11':4},'sleem' : {'2':3,'10':2,'11':4},'engy' : {'1':3,'2':5,'10':1}}
listt = sorted(data.items(), key=operator.itemgetter(1))
for k,v in listt:
	print(k,v)
dic={'ahmad':5,'medo':4,'eisso':1,'khalid':7}

listt = sorted(dic.items(), key=operator.itemgetter(5))
list_string = str(listt)

with open("list.txt", "w") as text_file:
	text_file.write(list_string)
with open("list.txt", "r") as text_file:
	eio = text_file.read()

list_again = ast.literal_eval(list_string)


print(listt)

eissa={}

for it in listt:
	eissa[it[0]]=it[1]

print(eissa)
