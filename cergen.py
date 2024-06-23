import random
import math
from typing import Union
import numpy as np
import time

RANDOM_INTEGER_OPTION = 0
RANDOM_FLOAT_OPTION = 1

NONE_TYPE = 0
SCALAR_TYPE = 1
VECTOR_TYPE = 2
MATRIX_TYPE = 3
TENSOR_TYPE = 4


#### private functions ####
def generate_random_numbers(element_number, minimum_value, maximum_value, number_type):
	number_list = []
	for i in range(element_number):
		if number_type == RANDOM_INTEGER_OPTION:
			number_list.append(random.randint(minimum_value, maximum_value))
		elif number_type == RANDOM_FLOAT_OPTION:
			number_list.append(random.uniform(minimum_value, maximum_value))
		else:
			raise Exception("Unknown number type")
	return number_list


def multiply_list_elements(lst):
	multiplication = 1
	for i in lst:
		multiplication *= i
	return multiplication


def nest_list(flat_list, dimensions):
	# base case
	if not dimensions:
		return flat_list[0]
	# recursive case
	group_size = dimensions[-1]
	resized_tensor = []
	i = 0
	while i < len(flat_list):
		resized_tensor.append(flat_list[i: i + group_size])
		i += group_size
	return nest_list(resized_tensor, dimensions[:-1])


def is_scalar(data):
	return not isinstance(data, list)


def is_vector(data):
	if (data is None) or (is_scalar(data)):
		return False
	for element in data:
		if not is_scalar(element):
			return False
	return True


def is_matrix(data):
	if (data is None) or (is_scalar(data)) or (is_vector(data)):
		return False
	for element in data:
		if not is_vector(element):
			return False
	return True


def is_tensor(data):
	if (data is None) or (is_scalar(data)) or (is_vector(data)) or (is_matrix(data)):
		return False
	return True


def get_data_type(data):
	if data is None:
		return NONE_TYPE
	if is_scalar(data):
		return SCALAR_TYPE
	if is_vector(data):
		return VECTOR_TYPE
	if is_matrix(data):
		return MATRIX_TYPE
	return TENSOR_TYPE


def get_data_dimensions(data):
	if data is None:
		raise Exception("Error: Wrong argument None")
	if is_scalar(data):
		return (0,)
	return get_non_scalar_data_dimensions(data)


def get_non_scalar_data_dimensions(data):
	# base case
	if type(data) != list:
		return []
	if len(data) == 0:
		return [0]
	# recursive case
	return [len(data)] + get_non_scalar_data_dimensions(data[0])


def my_div(x, y):
	if y == 0:
		raise ValueError("Error: Cannot divide by zero")
	return x / y


def outer_product(vector_1, vector_2):
	result = []
	for i in range(len(vector_1)):
		row = []
		for j in range(len(vector_2)):
			row.append(vector_1[i] * vector_2[j])
		result.append(row)
	return result


#### private functions ####


def cekirdek(number: int):
	random.seed(number)
	return


def rastgele_dogal(boyut, aralik=(0, 100), dagilim='uniform'):
	# check if distribution parameter is given correctly
	if dagilim != 'uniform':
		raise ValueError("Dagilim parameter must be uniform")
	# create a list of integers
	flat_data = generate_random_numbers(multiply_list_elements(boyut), aralik[0], aralik[1], RANDOM_INTEGER_OPTION)
	data = nest_list(flat_data, boyut)
	return gergen(data)


def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
	if dagilim != 'uniform':
		raise ValueError("Dagilim parameter must be uniform")
	# create a list of floats
	flat_data = generate_random_numbers(multiply_list_elements(boyut), aralik[0], aralik[1], RANDOM_FLOAT_OPTION)
	data = nest_list(flat_data, boyut)
	return gergen(data)


def handle_binary_operations_edge_cases(operand_1, operand_2, callback):
	if type(operand_2) == gergen and operand_2.get_tensor_type() == NONE_TYPE:
		raise ValueError("Error: Binary operation with gergen of type NONE")
	# check if this gergen is a scalar
	if operand_1.get_tensor_type() == NONE_TYPE:
		raise ValueError("Error: Binary operation with gergen of type NONE")
	if operand_1.get_tensor_type() == SCALAR_TYPE:
		if type(operand_2) != gergen:
			return gergen(callback(operand_1.get_veri(), operand_2))
		elif operand_2.get_tensor_type() == SCALAR_TYPE:
			return gergen(callback(operand_1.get_veri(), operand_2.get_veri()))
		else:
			raise ValueError("Exception")


def create_empty_nested_list(dimensions):
	result = []
	# base case
	if len(dimensions) == 1:
		for i in range(dimensions[0]):
			result.append(0)
		return result
	# recursive case
	for i in range(dimensions[0]):
		result.append(create_empty_nested_list(dimensions[1:]))
	return result


def get_nested_list_cell(source_list, path):
	current_list = source_list
	for index in path[:-1]:
		current_list = current_list[index]
	return current_list[path[-1]]


def set_nested_list_cell(data, target_list, path):
	current_list = target_list
	for index in path[:-1]:
		current_list = current_list[index]
	current_list[path[-1]] = data


def is_matrix_dimensions_suitable_for_multiplication(dimensions_1, dimensions_2):
	if dimensions_1[1] == dimensions_2[0]:
		return True
	return False


def vector_inner_product(vector1, vector2):
	result = 0
	for i in range(len(vector1)):
		result += vector1[i] * vector2[i]
	return result


def multiply_column_vectors(matrix_1, matrix_2):
	result = []
	for i in range(len(matrix_1)):
		row = []
		for j in range(len(matrix_2)):
			row.append(vector_inner_product(matrix_1[i], matrix_2[j]))
		result.append(row.copy())
	return result


def get_matrix_product_dimension(dimensions_1, dimensions_2):
	return tuple[dimensions_1[0], dimensions_2[1]]


def operate_binary_operation_helper(element_1, element_2, path, result, callback_function_1, callback_function_2):
	# base case (cell case)
	if type(element_1) != list:
		callback_function_1(callback_function_2(element_1, element_2), result, path)
		return
	# recursive case
	for sub_tensor_index in range(len(element_1)):
		if (type(element_2) != list):
			operate_binary_operation_helper(element_1[sub_tensor_index], element_2, path + [sub_tensor_index], result,
			                                callback_function_1, callback_function_2)
		else:
			operate_binary_operation_helper(element_1[sub_tensor_index], element_2[sub_tensor_index],
			                                path + [sub_tensor_index], result, callback_function_1, callback_function_2)


def operate_binary_operation(element_1, element_2, dimensions, callback_function_1, callback_function_2):
	result_tensor = create_empty_nested_list(dimensions)
	operate_binary_operation_helper(element_1, element_2, [], result_tensor, callback_function_1, callback_function_2)
	return gergen(result_tensor)


def operate_unary_operation(lst, dimensions, callback_function, *additional_callback_args):
	result_tensor = create_empty_nested_list(dimensions)
	operate_unary_operation_helper(lst, [], result_tensor, callback_function, additional_callback_args)
	return gergen(result_tensor)


def operate_unary_operation_helper(lst, path, result, callback_function, *additional_callback_args):
	# base case
	if type(lst) != list:
		set_nested_list_cell(callback_function(lst, additional_callback_args), result, path)
		return
	# recursive case
	for sub_tensor_index in range(len(lst)):
		operate_unary_operation_helper(lst[sub_tensor_index], path + [sub_tensor_index], result, callback_function,
		                               additional_callback_args)


def my_method(lst, dimensions, n):
	result_tensor = create_empty_nested_list(dimensions)
	my_method_helper(lst, [], result_tensor, n)
	return gergen(result_tensor)

def my_method_helper(lst, path, result, n):
	# base case
	if type(lst) != list:
		set_nested_list_cell(lst ** n, result, path)
		return
	# recursive case
	for sub_tensor_index in range(len(lst)):
		my_method_helper(lst[sub_tensor_index], path + [sub_tensor_index], result, n)

def subtract_from_nested_list(lst, element, dimensions):
	return operate_binary_operation(lst, element, dimensions, set_nested_list_cell, lambda x, y: x - y)


def add_scalar_to_nested_list(lst, number, dimensions):
	return operate_binary_operation(lst, number, dimensions, set_nested_list_cell, lambda x, y: x + y)


def add_nested_lists(lst1, lst2, dimensions):
	return operate_binary_operation(lst1, lst2, dimensions, set_nested_list_cell, lambda x, y: x + y)


def multiply_nested_list_by_scalar(lst, number, dimensions):
	return operate_binary_operation(lst, number, dimensions, set_nested_list_cell, lambda x, y: x * y)


def multiply_nested_lists(lst_1, lst_2, dimensions):
	return operate_binary_operation(lst_1, lst_2, dimensions, set_nested_list_cell, lambda x, y: x * y)


def divide_nested_list_by_scalar(lst, number, dimensions):
	return operate_binary_operation(lst, number, dimensions, set_nested_list_cell, lambda x, y: x / y)


def divide_nested_lists(lst_1, lst_2, dimensions):
	return operate_binary_operation(lst_1, lst_2, dimensions, set_nested_list_cell, lambda x, y: x / y)


def multiply_elements(lst):
	result = 1
	for element in lst:
		result *= element
	return result


def flatten(nested):
	# base case
	if len(nested) == 0:
		return []
	# recursive case
	elif type(nested[0]) != list:
		return [nested[0]] + flatten(nested[1:])
	return flatten(nested[0]) + flatten(nested[1:])


class gergen:
	__veri = None
	D = None
	__boyut = None
	
	def __init__(self, veri=None):
		self.__veri = veri
		self.__boyut = self.get_gergen_dimensions()
		if type(veri) != list:
			self.D = None
		else:
			self.D = create_empty_nested_list(list(reversed(self.__boyut)))
			self.assign_transposed_tensor()
	
	def __getitem__(self, index):
		if self.__veri is None:
			raise Exception("veri not defined")
		if is_scalar(self.__veri):
			raise Exception("veri is scalar which is not subscriptable")
		return self.__veri[index]
	
	# Generates a string representation
	def __str__(self):
		return str(self.__veri)
	
	def __rmul__(self, other: Union['gergen', int, float]) -> 'gergen':
		# here
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, lambda x, y: x * y)
		if edge_case_result is not None:
			return edge_case_result
		# here
		# check if the other is scalar
		if type(other) != gergen:
			return multiply_nested_list_by_scalar(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			print("Error: Cannot multiply two gergen with different dimension")
			exit(0)
		else:
			return multiply_nested_lists(self.__veri, other.__veri, self.boyut())
	
	def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
		# here
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, lambda x, y: x * y)
		if edge_case_result is not None:
			return edge_case_result
		# here
		# check if the other is scalar
		if type(other) != gergen:
			return multiply_nested_list_by_scalar(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			print("Error: Cannot multiply two gergen with different dimension")
			exit(0)
		else:
			return multiply_nested_lists(self.__veri, other.__veri, self.boyut())
	
	def __rtruediv__(self, other: Union['gergen', int, float]) -> 'gergen':
		if other == 0:
			raise ZeroDivisionError("Error: Cannot divide by zero")
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, my_div)
		if edge_case_result is not None:
			return edge_case_result
		# check if the other is scalar
		if type(other) != gergen:
			if (other == 0):
				raise ZeroDivisionError("Error: Cannot divide by zero")
			return divide_nested_list_by_scalar(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			raise ValueError("Error: Cannot divide two gergen with different dimension")
		else:
			return divide_nested_lists(self.__veri, other.__veri, self.boyut())
	
	def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
		if other == 0:
			raise ZeroDivisionError("Error: Cannot divide by zero")
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, my_div)
		if edge_case_result is not None:
			return edge_case_result
		# check if the other is scalar
		if type(other) != gergen:
			if (other == 0):
				raise ZeroDivisionError("Error: Cannot divide by zero")
			return divide_nested_list_by_scalar(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			raise ValueError("Error: Cannot divide two gergen with different dimension")
		else:
			return divide_nested_lists(self.__veri, other.__veri, self.boyut())
	
	"""
		Defines the addition operation for gergen objects.
		Called when a gergen object is added to another, using the '+' operator.
		The operation is element-wise.
	"""
	
	def __radd__(self, other: Union['gergen', int, float]) -> 'gergen':
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, lambda x, y: x + y)
		if edge_case_result is not None:
			return edge_case_result
		# check if the other is scalar
		if type(other) != gergen:
			return add_scalar_to_nested_list(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			raise ValueError("Error: Cannot add two gergen with different dimension")
		else:
			return add_nested_lists(self.__veri, other.__veri, self.boyut())
	
	def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, lambda x, y: x + y)
		if edge_case_result is not None:
			return edge_case_result
		# check if the other is scalar
		if type(other) != gergen:
			return add_scalar_to_nested_list(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			raise ValueError("Error: Cannot add two gergen with different dimension")
		else:
			return add_nested_lists(self.__veri, other.__veri, self.boyut())
	
	def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
		edge_case_result = None
		edge_case_result = handle_binary_operations_edge_cases(self, other, lambda x, y: x - y)
		if edge_case_result is not None:
			return edge_case_result
		# check if the other is scalar
		if type(other) != gergen:
			return subtract_from_nested_list(self.__veri, other, self.boyut())
		# check if the gergens have the same dimensions
		if self.boyut() != other.__boyut:
			raise ValueError("Error: Cannot subtract two gergen with different dimension")
		return subtract_from_nested_list(self.__veri, other.__veri, self.boyut())
	
	# Returns the total number of elements in the gergen
	def uzunluk(self):
		if self.__veri is None:
			raise ValueError("Error: Uzunluk is not defined for gergen of type NONE")
		if self.get_tensor_type() == SCALAR_TYPE:
			return 1
		return len(flatten(self.__veri))
	
	# Returns the shape of the gergen
	def boyut(self):
		return self.__boyut
	
	# Returns the transpose of gergen
	def devrik(self):
		return gergen(self.D)
	
	# Calculates the sine of each element in the given `gergen`.
	def sin(self):
		edge_case_result = None
		edge_case_result = self.handle_unary_operations_edge_cases(lambda x, *args: math.sin(x))
		if edge_case_result is not None:
			return edge_case_result
		return operate_unary_operation(self.__veri, self.boyut(), lambda x, *args: math.sin(x))
	
	# Calculates the cosine of each element in the given `gergen`.
	def cos(self):
		edge_case_result = None
		edge_case_result = self.handle_unary_operations_edge_cases(lambda x, *args: math.cos(x))
		if edge_case_result is not None:
			return edge_case_result
		return operate_unary_operation(self.__veri, self.boyut(), lambda x, *args: math.cos(x))
	
	# Calculates the tangent of each element in the given `gergen`.
	# TODO: what if the element is 90 degrees? The result is infinity. Check this case!
	def tan(self):
		edge_case_result = None
		edge_case_result = self.handle_unary_operations_edge_cases(lambda x, *args: math.tan(x))
		if edge_case_result is not None:
			return edge_case_result
		return operate_unary_operation(self.__veri, self.boyut(), lambda x, *args: math.tan(x))
	
	# Raises each element of the gergen object to the power 'n'. This is an element-wise operation.
	def us(self, n: int):
		edge_case_result = self.handle_unary_operations_edge_cases(lambda x, *args: x ** args[0][0], n)
		if edge_case_result is not None:
			return edge_case_result
		return my_method(self.__veri, self.boyut(), n)
	
	# Applies the logarithm function to each element of the gergen object, using the base 10.
	def log(self):
		custom_function = lambda x, *args: math.log10(x)
		edge_case_result = self.handle_unary_operations_edge_cases(custom_function)
		if edge_case_result is not None:
			return edge_case_result
		return operate_unary_operation(self.__veri, self.boyut(), custom_function)
	
	def ln(self):
		edge_case_result = self.handle_unary_operations_edge_cases(lambda x, y: math.log(x))
		if edge_case_result is not None:
			return edge_case_result
		return operate_unary_operation(self.__veri, self.boyut(), lambda x, y: math.log(x))
	
	def L1(self):
		flattened_list = flatten(self.__veri)
		l1_norm = 0
		for element in flattened_list:
			l1_norm += abs(element)
		return l1_norm
	
	def L2(self):
		flattened_list = flatten(self.__veri)
		l2_norm = 0
		for element in flattened_list:
			l2_norm += element ** 2
		return math.sqrt(l2_norm)
	
	def Lp(self, p):
		if (p <= 0) or (not isinstance(p, int)):
			print("Error: p must be an integer greater than 0")
			exit(0)
		flattened_list = flatten(self.__veri)
		lp_norm = 0
		for element in flattened_list:
			lp_norm += abs(element) ** p
		return lp_norm ** (1 / p)
	
	def listeye(self):
		# Converts the gergen object into a list or a nested list, depending on its dimensions.
		if self.get_tensor_type() == NONE_TYPE:
			return []
		if self.get_tensor_type() == SCALAR_TYPE:
			return [self.__veri]
		else:
			return self.__veri
	
	def duzlestir(self):
		if self.get_tensor_type() == NONE_TYPE:
			raise ValueError("Gergen is None")
		elif self.get_tensor_type() == SCALAR_TYPE:
			return [self.__veri]
		return gergen(flatten(self.__veri))
	
	# Reshapes the gergen object to a new shape 'yeni_boyut', which is specified as a tuple.
	def boyutlandir(self, yeni_boyut):
		if self.__veri is None:
			raise ValueError("Boyutlandir is not defined for None gergens")
		if is_scalar(self.__veri):
			raise ValueError("Error: Boyutlandir is not defined for scalar gergens")
		# check if new dimensions is given as a tuple
		if (not isinstance(yeni_boyut, tuple)):
			raise ValueError("Error: yeni_boyut must be of type tuple")
		flattened_list = flatten(self.__veri)
		# check if the dimensions is applicable to tensor
		if multiply_elements(yeni_boyut) != len(flattened_list):
			raise ValueError("Error: yeni_boyut does not match the number of elements in tensor")
		return nest_list(flattened_list, list(yeni_boyut))
	
	# Calculates the inner (dot) product of this gergen object with another.
	def ic_carpim(self, other):
		if not isinstance(other, gergen):
			raise ValueError("Error: ic_carpim must be of type gergen")
		if self.get_tensor_type() == NONE_TYPE or other.get_tensor_type() == NONE_TYPE:
			raise TypeError("Error: ic_carpim operation using none gergens")
		if self.get_tensor_type() == SCALAR_TYPE or other.get_tensor_type() == SCALAR_TYPE:
			raise ValueError("Error: ic_carpim operation using scalar gergens")
		if self.get_tensor_type() != other.get_tensor_type():
			raise ValueError("Error: ic_carpim operation using different tensor dimensions")
		if self.get_tensor_type() == VECTOR_TYPE:
			if self.boyut() != other.boyut():
				raise ValueError("Error: Boyut's of operands of ic_carpim are not equal")
			return sum(operate_binary_operation(self.__veri, other.__veri, self.boyut(), set_nested_list_cell,
			                                    lambda x, y: x * y).get_veri())
		if self.get_tensor_type() == MATRIX_TYPE:
			if not is_matrix_dimensions_suitable_for_multiplication(self.boyut(), other.boyut()):
				raise ValueError("Error: ic_carpim operation using inappropriate matrix dimensions")
			return multiply_column_vectors(self.__veri, other.devrik().get_veri())
	
	def dis_carpim(self, other):
		if not isinstance(other, gergen):
			raise TypeError("Error: dis_carpim must be of type gergen")
		if self.get_tensor_type() == NONE_TYPE or other.get_tensor_type() == NONE_TYPE:
			raise ValueError("Error: dis_carpim operation using none gergens")
		if self.get_tensor_type() == SCALAR_TYPE or other.get_tensor_type() == SCALAR_TYPE:
			raise ValueError("Error: dis_carpim operation using scalar gergens")
		if self.get_tensor_type() != VECTOR_TYPE or other.get_tensor_type() != VECTOR_TYPE:
			raise ValueError("Error: dis_carpim operation using non-vectors")
		if self.boyut() != other.boyut():
			raise ValueError("Error: Boyut's of operands of dis_carpim are not equal")
		return outer_product(self.__veri, other.get_veri())
	
	def topla(self, eksen=None):
		if not (type(eksen) == int or eksen is None):
			raise TypeError("eksen must be an integer or None.")
		if eksen is None:
			return sum(flatten(self.__veri))
		if not (eksen < len(self.boyut())):
			raise ValueError("the specified eksen is out of bounds")
		temp = []
		for i in range(eksen):
			pass
	
	def ortalama(self, eksen=None):
		# Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
		pass
	
	#### private methods ####
	def get_gergen_dimensions(self):
		if self.__veri is None:
			return None
		return tuple(get_data_dimensions(self.__veri))
	
	def set_transpose_cell(self, data, path):
		set_nested_list_cell(data, self.D, path)
	
	def set_cell(self, data, path):
		set_nested_list_cell(data, self.__veri, path)
	
	def get_cell(self, path):
		get_nested_list_cell(self.__veri, path)
	
	def get_veri(self):
		return self.__veri
	
	def assign_transposed_tensor(self):
		self.traverse_tensor_cells(self.__veri, [], self.set_transpose_cell, [])
	
	def traverse_tensor_cells(self, tensor, tensor_path, callback_function, *callback_args):
		# base case (scalar / cell case)
		if type(tensor) != list:
			callback_function(tensor, list(reversed(tensor_path)))
			return
		# recursive case
		for sub_tensor_index in range(len(tensor)):
			self.traverse_tensor_cells(tensor[sub_tensor_index], tensor_path + [sub_tensor_index], callback_function)
	
	def traverse_tensor_cells_simultaneously(self, tensor_1, tensor_2, tensor_path, callback_function, *callback_args):
		# base case (scalar / cell case)
		if type(tensor_1) != list:
			callback_function(tensor_1, tensor_2, tensor_path)
			return
		# recursive case
		for sub_tensor_index in range(len(tensor_1)):
			self.traverse_tensor_cells_simultaneously(tensor_1[sub_tensor_index], tensor_2[sub_tensor_index],
			                                          tensor_path + [sub_tensor_index], callback_function,
			                                          callback_args)
	
	def handle_unary_operations_edge_cases(self, callback, *additional_callback_args):
		if self.get_tensor_type() == NONE_TYPE:
			raise ValueError("Error: Operation with gergen of type none")
		if self.get_tensor_type() == SCALAR_TYPE:
			return gergen(callback(self.__veri, additional_callback_args))
	
	def get_tensor_type(self):
		return get_data_type(self.__veri)



def example_1():
	# Example 1
	boyut = (64, 64)
	gergen_1 = rastgele_gercek(boyut)
	gergen_2 = rastgele_gercek(boyut)
	
	start = time.time()
	# TODO
	gergen_1.ic_carpim(gergen_2)
	end = time.time()
	
	a = np.random.rand(64, 64)
	b = np.random.rand(64, 64)
	start_np = time.time()
	# Apply the same equation for NumPy equivalent
	aT = a.transpose()
	np_result = np.dot(aT, b)
	end_np = time.time()
	
	# TODO:
	# Compare if the two results are the same
	# Report the time difference
	print("Time taken for gergen:", end - start)
	print("Time taken for numpy:", round(end_np - start_np, 5))
	
	"""
	Time taken for gergen: 0.021999835968017578
	Time taken for numpy: 0.0

	Result for numpy is significantly less than gergen class.
	Probably numpy is more efficient in calculations.
	"""


def example_2():
	# Example 2
	# TODO:
	BOYUT = (4, 16, 16, 16)
	A = rastgele_gercek(BOYUT)
	B = rastgele_gercek(BOYUT)
	C = rastgele_gercek(BOYUT)
	
	# gergen time
	start = time.time()
	
	a_times_b = A.__mul__(B)
	c_times_a = C.__mul__(A)
	b_times_c = B.__mul__(C)
	
	ab_plus_ca = a_times_b.__add__(c_times_a)
	list_result = ab_plus_ca.__add__(b_times_c)
	
	gergen_result = list_result.ortalama()
	
	end = time.time()
	
	a = np.random.rand(4, 16, 16, 16)
	b = np.random.rand(4, 16, 16, 16)
	c = np.random.rand(4, 16, 16, 16)
	# numpy time
	start_np = time.time()
	# Apply the same equation for NumPy equivalent
	axb = np.multiply(a, b)
	cxa = np.multiply(c, a)
	bxc = np.multiply(b, c)
	
	axbcxa = np.add(axb, cxa)
	matrix_result = np.add(axbcxa, bxc)
	
	matrix_result.mean()
	
	end_np = time.time()
	
	print("Time taken for gergen:", end - start)
	print("Time taken for numpy:", end_np - start_np)
	
	"""
	Time taken for gergen: 0.1483149528503418
	Time taken for numpy: 0.0021309852600097656

	Result for numpy is significantly less than gergen class.
	Probably numpy is more efficient in calculations.
	Also, I could not implement method ortalama

	"""
	
	return gergen_result


def example_3():
	# Example 3
	# TODO:
	boyut = (3, 64, 64)
	A = rastgele_gercek(boyut)
	B = rastgele_gercek(boyut)
	
	start = time.time()
	# TODO
	# Apply given equation
	sinA = A.sin()
	cosB = B.cos()
	sinAcosB = sinA.__add__(cosB)
	
	lnned = sinAcosB.ln()
	squared = lnned.us(2)
	
	gergen_result = squared.__truediv__(8)
	
	end = time.time()
	
	a = np.random.rand(3, 64, 64)
	b = np.random.rand(3, 64, 64)
	start_np = time.time()
	# Apply the same equation for NumPy equivalent
	sina = np.sin(a)
	cosb = np.cos(b)
	sinacosb = np.add(sina, cosb)
	
	lnnednp = np.log(sinacosb)
	
	squarednp = np.square(lnnednp)
	
	np_result = np.divide(squarednp, 8)
	
	end_np = time.time()
	# TODO:
	# Compare if the two results are the same
	# Report the time difference
	print("Time taken for gergen:", end - start)
	print("Time taken for numpy:", end_np - start_np)
	
	"""
	Time taken for gergen: 0.12713003158569336
	Time taken for numpy: 0.0011098384857177734

	Again, numpy is very successful in computing.
	This is caused of efficiency issues in my implementation.
	"""
	return gergen_result


if __name__ == '__main__':
	
	example_1()
	example_2()
	example_3()
	