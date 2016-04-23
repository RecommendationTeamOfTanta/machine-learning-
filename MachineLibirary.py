import csv
import random
import math
from math import sqrt
import operator
import array
import itertools
import ast
import io
##################################################################
#split to train and test
##################################################################
def	split_data(data,prob):
#split data into fractions [prob, 1 - prob]"""
	results = [],[]
	for	row	in	data:
		results[0 if random.random() < prob else 1].append(row)
	return results

##################################################################
#split to train and test
##################################################################
def	train_test_split(x,y,test_pct):
	#pair corresponding values
	data = zip(x,y)				
	#split the data set of pairs
	train,test = split_data(data,1 - test_pct)		
	#magical un-zip trick
	x_train,y_train = zip(*train)						
	x_test,y_test = zip(*test)
	return x_train,x_test,y_train,y_test



##################################################################
#load datasets and prepare training and testing set
##################################################################
def loadDataset(filename, split, training_set={}, test_set={},all_data_set={},training_tuple=[],test_tuple=[],all_data_set_tuple=[]):
	train_prepair = []
	test_prepair = []

	with open(filename)  as csvfile:
		lines = csv.reader(csvfile)
	    #to separate header of data
		headers = next(lines)
		#column = {}

		#prepare the traning set to be dictionary of lists
		for h in headers:
			training_set[h] = list()
			test_set[h] = list()
			all_data_set[h] = list()

		dataset = list(lines)

		for x in range(len(dataset) - 1):
			all_data_set
			for y in range(len(headers)):

				#to skip the data column in dataset kc_house_data.csv
				if y == 1:continue
				dataset[x][y] = float(dataset[x][y])

			
			#devide data to train and test
			if random.random() < split:
				train_prepair.append(dataset[x])
			else:test_prepair.append(dataset[x])
		
			#fill all_data_set
		for row in dataset:
			for h,v in zip(headers,row):
				all_data_set[h].append(v)

		#fill training_set
		for row in train_prepair:
			for h,v in zip(headers,row):
				training_set[h].append(v)
		
		#fill test_set
		for row in test_prepair:
			for h,v in zip(headers,row):
				test_set[h].append(v)

		for k,v in all_data_set.items():
			all_data_set_tuple.append((k,v))

		for   k,v in training_set.items():
			training_tuple.append((k,v))

		for k,v in test_set.items():
			test_tuple.append((k,v))



###################################################
#euclidean similarity
###################################################
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


#################################################
#k-nearest neighbours
#################################################	
def getNeighbors(training_set, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(training_set)):
		dist = euclideanDistance(testInstance, training_set[x], length)
		distances.append((training_set[x],dist))

	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	eissa = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors



##################################################
#return the magority one of neighbour
##################################################
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

######################################################
#calculate the accuracy
######################################################
def getAccuracy(test_set, predictions):
	correct = 0
	for x in range(len(test_set)):
		if test_set[x][-1] == predictions[x]:
			correct += 1
	return (correct / float(len(test_set))) * 100.0


############################################################
#linear regression to solve intercept and slope analatically
############################################################
def linear_regression(input_feature, output):
	n = len(input_feature)
	# compute the mean of input_feature and output
	xi_sum = sum(input_feature)
	yi_sum = sum(output)
	# compute the product of the output and the input_feature and its mean
	xiyi = [a * b for a,b in zip(input_feature,output)]
	xiyi_sum = sum(xiyi)
	# compute the squared value of the input_feature and its mean
	xixi = [a * b for a,b in zip(input_feature,input_feature)]
	xixi_sum = sum(xixi)
	# use the formula for the slope
	slope = (xiyi_sum - xi_sum * yi_sum / n) / (xixi_sum - xi_sum * xi_sum / n)
	# use the formula for the intercept
	intercept = yi_sum / n - slope * xi_sum / n
	return (intercept, slope)


#Prediction functions
def prediction_of_regression(input,slope,intercept):
	predicted_output = intercept + slope * input
	return predicted_output

#get inverse of regression
def inverse_regression(output,slope,intercept):
	predicted_input = (output - intercept) / slope
	return predicted_input

#########################
#Residual sum of squares
#########################
def residual_sum_of_squares(input,output,intercept,slope):
	predicted = [prediction_of_regression(i,slope,intercept) for i in input]
	residuals = [a - b for a, b in zip(predicted, output)]
	rere = [a * b for a,b in zip(residuals,residuals)]
	rss = sum(rere)
	return rss



###########################################
###Multible regression
###########################################

#features=[list of features],output=<output column name>
def prepare_data_dic(data,features,output):
	features_matrix_dic = {}
	for feature in features:
			features_matrix_dic[feature] = list()
			for i in data[feature]:
				features_matrix_dic[feature].append(i)
	features_matrix_dic['constant'] = [1 for i in data]
	output_vector = data[output]
	return(features_matrix_dic,output_vector) 

def prepare_data_tuple(data,features,output):
	features_matrix_tuple = []

	features_matrix_tuple.append(('constant',[1 for i in range(len(data[0][1]))]))

	for feature in features:
		features_matrix_tuple.append((feature,get_column(data,feature)))
			
	output_vector = get_column(data,output)

	return(features_matrix_tuple,output_vector)


#the WT*x model(w0x0 + w1X1 + W2X2 + ........+WnXn)
def predict_output(feature_matrix, weights):
	predictions = dot_product_m_v(feature_matrix,weights)
	return predictions


def feature_derivative(errors,feature):
	derivative = 2 * dot_product_v_v(errors, feature)
	return (derivative)

#gradient descent algorithm
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
	coverged = False
	while not coverged:
		predictions = predict_output(feature_matrix,initial_weights)
		error = [a - b for a,b in zip(predictions,output)]
		gradient_sum_squares = 0
		for i in range(len(initial_weights)):
			feature = feature_matrix[i][1]
			derivative = feature_derivative(error,feature)

			gradient_sum_squares += derivative * derivative
			initial_weights[i]-= step_size * derivative
			gradient_magnitude = sqrt(gradient_sum_squares)
			if gradient_magnitude < tolerance:
				coverged = True
	return(initial_weights)


######################
#Utilities functions
######################

#get a list of first element in a dictionary of lists
def index_in_dic(dic,index_of_list=0):
	lista = list()
	for it in dic:
		lista.append(dic[it][index_of_list])
	return lista

#return a row from tuple by index
def index_in_tuple(tuple,index_of_list=0):
	lista = list()
	for it in tuple:
		lista.append(it[1][index_of_list])
	return lista

#get a list of first element in a list of tuples
def first_in_list_tuples(lista_of_tuples):
	lista = list()
	for item in lista_of_tuples:
		list.append(item[1][0])
	return lista

#convert dictionary of lists to list of tuples
def dic_to_tuple(dic):

	return [(key,value) for key,value in dic.items()]

#convert list of tubles to dictionary
def tuple_to_dic(tuple):
	for item in tuple:
		return {key:value for key,value in tuple}


#dot product of matrix and vector
def dot_product_m_v(matrix,vector):
	lista = list()
	for i in range(len(matrix[0][1])):
		lista.append(sum([a * b for a,b in zip(index_in_tuple(matrix,i),vector)]))
	return lista

def dot_product_v_v(vector1,vector2):
	return sum(a * b for a,b in zip(vector1,vector2))


#get column of the data
def get_column(lista,column):
	for tuple in lista:
		if tuple[0].startswith(column):
			return tuple[1]





dictionary = {'1to5':[1,2,3,4,5],'6to10':[6,7,8,9,10],'11to15':[11,12,13,14,15]}

list_of_tuple = [('1to5',[1,2,3,4,5]),('6to10',[6,7,8,9,10]),('11to15',[11,12,13,14,15])]

eissa = get_column(list_of_tuple,'6to10')
eisis = dic_to_tuple(dictionary)



######################################################
#Testing
######################################################
e = 4e-12
r = (2 * e)
input = [1,2,3,4,5]
output = [1 + 1 * a for a in input]
(intercept,slope) = linear_regression(input,output)

print("intercept: " + str(intercept))
print("slope: " + str(slope))

training_set = {}
test_set = {}
all_data_set = {}
training_tuple = [] 
test_tuple = []
all_data_set_tuple = []
										
loadDataset("kc_house_data.csv",.8,training_set,test_set,all_data_set,training_tuple,test_tuple,all_data_set_tuple)

#eeff =
#dot_product(get_column(training_tuple,'sqft_living'),get_column(training_tuple,'price'))
faetures_matrix_dic,output_vector = prepare_data_dic(all_data_set,['sqft_living'],'price')
faetures_matrix_tuple,output_vector_tuple = prepare_data_tuple(training_tuple,['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_living15'],'price')


#gradient test
gradient_weights = regression_gradient_descent(faetures_matrix_tuple,output_vector_tuple,[-100000,1,1,1,1, 1],4e-12,1e9)

out = output_vector[0]
test_feature_matrix, test_output = prepare_data_tuple(test_tuple,['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_living15'], 'price')
predictions_from_latestModel = predict_output(test_feature_matrix, gradient_weights)

eissas = predictions_from_latestModel[0]
outt = test_output[0]
outt = test_output[1]


eissasasddss = (dot_product_m_v(faetures_matrix_tuple,[0,0]))




test_prediction = predict_output(faetures_matrix_tuple,[0,0])
errors = [a - b for a,b in zip(test_prediction,output_vector_tuple)]
feature = get_column(faetures_matrix_tuple,'constant')
derivative = feature_derivative(errors,feature)


eissa = predict_output(faetures_matrix_tuple,[1,1,1])

dddd = sum([float(i) for i in output_vector_tuple])


#intercept, slope = linear_regression([float(i) for i in
#training_set['sqft_living']],[float(i) for i in training_set['price']])
intercept, slope = linear_regression(training_set['sqft_living'], training_set['price'])

print("intercept: " + str(intercept))
print("slope: " + str(slope))

#test if the data devision work correctly
total = len(training_set['price']) + len(test_set['price'])
testo = (len(test_set['price']) / total) * 100
train = (len(training_set['price']) / total) * 100

print("predicted: " + str(prediction_of_regression(test_set['sqft_living'][3],slope,intercept)))

print("actual: " + str(test_set['price'][3]))


rss = residual_sum_of_squares(training_set['sqft_living'],training_set['price'],intercept,slope)
print(str(rss))


#with open("list.txt", "w") as text_file:
#	text_file.write(str(training_set))

#with open("list1.txt", "w") as text_file:
#	text_file.write(str(test_set))


#with open("list1.txt", "r") as text_file:
#	eio = text_file.read()
#dic_again = ast.literal_eval(eio)
#eissaaa = dic_again['price']




