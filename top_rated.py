import productVectors as p_vector

def mean_item_based(data,rest_name):
	
	for rest in data:
		if rest==rest_name:
			rest_data = data[rest_name]
			summ = sum(rest_data.values())
			count = len(rest_data)
			avg = summ/count	
	return (rest_name,avg)


#testing
mean = mean_item_based(p_vector.transformed_data_Set,"granny")

print(mean)