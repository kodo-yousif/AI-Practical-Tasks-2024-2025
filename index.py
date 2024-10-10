import numpy as np

oneD_ZeroArray = np.zeros(10)
# make a list of zeros

oneD_OneArray = np.ones(10)
# make a list of ones 

oneD_RadnomArray = np.random.randint(1, 51, 10)
# startRange, EndRange, Number of elements

oneD_to10Array = np.arange(0, 10)
# start , end

oneD_reversedTo10Array = oneD_to10Array[::-1]
# start : end : steps

matrix_random = np.random.randint(1, 10, (3, 3))
# startRange, endRange, () this means a  N dimensional matrix depend on number of Commas

matrix_identity = np.eye(4)
# make a identity matrix of 4x4

array = np.linspace(0, 1, 5)
# a list with - startRange, EndRange, Number of Elements

x = np.arange(0, 100)
matrix_tenByTen = x.reshape(10, 10)
# you can change your list into a matrix but the product of the dimensions must be equal to the size of the original list

y = np.arange(10)
y_sum = np.sum(y)
y_mean = np.mean(y)
y_standardDevision = np.std(y)
# getting the sum and mean and standard division of a list

# print(y[4])
# print(y[0:3])
# print(y[-2:])
# start by default is 0, end by default is the end of the array, steps by default is 1

z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
p = z[(0), :]
# which row to include you can right as much you want even you can repeat the rows (0,0,0)

rows, cols = z.shape
# to know the size of rows and cols

top_left = z[0, 0]
top_right = z[0, cols - 1]
bottom_left = z[rows - 1, 0]
bottom_right = z[rows - 1, cols - 1]
# this is for getting the edges of the matrix

e = np.arange(9)
matrix = e.reshape(3, 3)
# array = matrix.flatten()
array = matrix.ravel()
# both method used to make a multi dimensional list to a one Dimensional list but :
# flatten will make the copy of the original array and modify the copy
# ravel will modify the original array 

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
concatenated_array = np.concatenate((array1, array2))
stacked_array_vertical = np.vstack((array1, array2))
stacked_array_horizontal = np.hstack((array1, array2))

# stacked_array_vertical = np.concatenate((array1, array2), axis=0)
# axis=0 is vertical

# stacked_array_horizontal = np.concatenate((array1, array2), axis=1)
# axis=1  is horizontal

sumOfArrays = array1 + array2
# result = np.add(array1, array2)

multiplyOfArrays = array1 * array2
# result = np.multiply(array1, array2)

dot_product = array1 @ array2
# dot_product = np.dot(array1, array2)

sqrt_array = np.sqrt(array1)
# square root over every element in the matrix

cubed_matrix = array1 ** 3
# raise every element to the power of 3

random_array = np.random.randint(0, 100, 10)
max_value = np.max(random_array)
min_value = np.min(random_array)
# get the minimum and maximum value in the matrix

indexOfMinValue = np.argmax(random_array)
indexOfMaxValue = np.argmin(random_array)
# get the indexes of the minimum and maximum value in the matrix

result_any = np.any(random_array > 50)
# boolean value that tells if there's any element that passes this condition

result_all = np.all(random_array < 50)
# boolean value that tells if every element of this list pass this condition

scalar = 10
addArraywithScalar = random_array + scalar
multiplyArraywithScalar = random_array * scalar
comparedArray = array1 > 5

mean_value = np.mean(array1)
arrayWithSubtractedMean = random_array - mean_value
