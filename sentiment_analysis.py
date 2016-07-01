import MachineLibirary as ml



# get the frequncy of a key in a tuple
def key_count_tuple(tuple,key):
	filtered_tuple = [(k,v) for k,v in tuple if key in k]
	return len(filtered_tuple)

#update a value in (key value) pairs in a list of tuples
def update_in_alist(alist, key, value):
	return [(k,v) if (k != key) else (key, value) for (k, v) in alist]



# count word frequencies in a sentence
def word_count(review_sentence):
	dictionary = {}
	list_of_tuples = []
	#split sentences into words
	temp_list = review_sentence.split()
	for word in temp_list:
		if word not in dictionary:
			dictionary[word] = 0 
		dictionary[word]+=1

	#convert dictionary into list of tuples
	for k,v in dictionary.items():
		list_of_tuples.append((k,v))
	return dictionary,list_of_tuples

# count word frequencies in a list and put the count in another column
def word_count_in_column(column,new_column):
	temp_list=[]
	temp_dic={}
	for ele in column:
		temp_dic[ele]=word_count(ele)[0]
		temp_list.append(word_count(ele)[1])
	return temp_dic,(new_column,temp_list)




#function to add abinary column to the data
def add_binary_column(dic,new_column,target_column,condition):
	cond = list(condition)
	if isinstance(dic,dict):
		if cond[0] == '=':
			dic[new_column] = [1 if float(i) == float(cond[1])  else 0 for i in dic[target_column]]
		if cond[0] == '>':
			dic[new_column] = [1 if float(i) > float(cond[1])  else 0 for i in dic[target_column]]
		if cond[0] == '<':
			dic[new_column] = [1 if float(i) < float(cond[1])  else 0 for i in dic[target_column]]
		if cond[0] == '>=':
			dic[new_column] = [1 if float(i) >= float(cond[1])  else 0 for i in dic[target_column]]
		if cond[0] == '<=':
			dic[new_column] = [1 if float(i) <= float(cond[1])  else 0 for i in dic[target_column]]
		if cond[0] == '!=':
			dic[new_column] = [1 if float(i) != float(cond[1])  else 0 for i in dic[target_column]]
	if isinstance(dic,list):
		if cond[0] == '=':
			dic.append((new_column,[1 if float(i) == float(cond[1])  else 0 for i in ml.get_column(dic,target_column)]))
		if cond[0] == '>':
			dic.append((new_column,[1 if float(i) > float(cond[1])  else 0 for i in ml.get_column(dic,target_column)]))
		if cond[0] == '<':
			dic.append((new_column,[1 if float(i) < float(cond[1])  else 0 for i in ml.get_column(dic,target_column)]))
		if cond[0] == '>=':
			dic.append((new_column,[1 if float(i) >= float(cond[1])  else 0 for i in ml.get_column(dic,target_column)]))
		if cond[0] == '<=':
			dic.append((new_column,[1 if float(i) <= float(cond[1])  else 0 for i in ml.get_column(dic,target_column)]))
		if cond[0] == '!=':
			dic.append((new_column,[1 if float(i) != float(cond[1])  else 0 for i in ml.get_column(dic,target_column)]))
	return dic


#testinggggggggggggggg
lista = []
listo = []
toto = "hello mr eissa hello"
listo = toto.split()
eisssa = {}







#dic = {"eissa":5,"tot":4,"soso":3,"tato":4}
training_set = {}
test_set = {}
all_data = {}
training_tuple = []
test_tuple = []
all_tuple = []

ml.loadDataset('amazon_baby.csv',.7,training_set,test_set,all_data,training_tuple,test_tuple,all_tuple)



#add sentiment column
modified_data = add_binary_column(all_tuple,'sentiment','rating','>3')

#get the review column
reviews = ml.get_column(modified_data,'review')




# add word_count column
modified_data.append(word_count_in_column(reviews,'word_count')[0])

fff = ""


