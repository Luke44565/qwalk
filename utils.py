import numpy as np
import math
import copy

class Polynomial:
	"""Class representing a polynomial with integer coefficients."""
	
	def __init__(self, coeffs: list[int]):
		"""Initialize the polynomial with a list of coefficients.
		
		coeffs[i] is the coefficient for X^i.
		"""
		self.coeffs = coeffs  # List of coefficients

	def degree(self) -> int:
		"""Returns the degree of the polynomial."""
		return len(self.coeffs) - 1

	def __add__(self, other):
		"""Adds two polynomials."""

		if type(other) is not Polynomial:
			other = Polynomial([other])

		max_len = max(len(self.coeffs), len(other.coeffs))
		result = [0] * max_len
		
		for i in range(max_len):
			coeff1 = self.coeffs[i] if i < len(self.coeffs) else 0
			coeff2 = other.coeffs[i] if i < len(other.coeffs) else 0
			result[i] = coeff1 + coeff2
		
		return Polynomial(result)
	
	def __radd__(self, other):
		"""Adds two polynomials."""

		if type(other) is not Polynomial:
			other = Polynomial([other])

		max_len = max(len(self.coeffs), len(other.coeffs))
		result = [0] * max_len
		
		for i in range(max_len):
			coeff1 = self.coeffs[i] if i < len(self.coeffs) else 0
			coeff2 = other.coeffs[i] if i < len(other.coeffs) else 0
			result[i] = coeff1 + coeff2
		
		return Polynomial(result)

	def __sub__(self, other):
		"""Subtracts two polynomials."""

		if type(other) is not Polynomial:
			other = Polynomial([other])

		max_len = max(len(self.coeffs), len(other.coeffs))
		result = [0] * max_len
		
		for i in range(max_len):
			coeff1 = self.coeffs[i] if i < len(self.coeffs) else 0
			coeff2 = other.coeffs[i] if i < len(other.coeffs) else 0
			result[i] = coeff1 - coeff2
		
		return Polynomial(result)

	def __rsub__(self, other):
		"""Subtracts two polynomials."""

		if type(other) is not Polynomial:
			other = Polynomial([other])

		max_len = max(len(self.coeffs), len(other.coeffs))
		result = [0] * max_len
		
		for i in range(max_len):
			coeff1 = self.coeffs[i] if i < len(self.coeffs) else 0
			coeff2 = other.coeffs[i] if i < len(other.coeffs) else 0
			result[i] = coeff2 - coeff1
		
		return Polynomial(result)

	def __mul__(self, other):
		"""Multiplies two polynomials."""

		if type(other) is not Polynomial:
			other = Polynomial([other])

		deg1 = len(self.coeffs)
		deg2 = len(other.coeffs)
		result_deg = deg1 + deg2 - 1
		result = [0] * result_deg
		
		for i in range(deg1):
			for j in range(deg2):
				result[i + j] += self.coeffs[i] * other.coeffs[j]
		
		return Polynomial(result)
	
	def __rmul__(self, other):
		"""Multiplies two polynomials."""

		if type(other) is not Polynomial:
			other = Polynomial([other])

		deg1 = len(self.coeffs)
		deg2 = len(other.coeffs)
		result_deg = deg1 + deg2 - 1
		result = [0] * result_deg
		
		for i in range(deg1):
			for j in range(deg2):
				result[i + j] += self.coeffs[i] * other.coeffs[j]
		
		return Polynomial(result)
	
	def __str__(self):
		terms = []
		for i, coeff in enumerate(self.coeffs):
			if coeff != 0:
				if i == 0:
					terms.append(f"{format_complex(coeff)}")
				elif i == 1:
					if coeff == 1:
						terms.append("x")
					else:
						terms.append(f"{format_complex_coeff(coeff)}x")
				else:
					if coeff == 1:
						terms.append(f"x^{i}")
					else:
						terms.append(f"{format_complex_coeff(coeff)}x^{i}")
		return " + ".join(terms) if terms else "0"
	
	def __repr__(self):
		return f"Poly({self.__str__()})"
	
	def __eq__(self, other):
		if type(other) is not Polynomial:
			return len(self.coeffs) == 1 and self.coeffs[0] == other
		else:
			return self.coeffs == other.coeffs


def format_real_coeff(num):
	if int(num) == round(num, 3):
		num = int(num)
		if num == 1:
			return ""
		elif num == -1:
			return "-"
		else:
			return str(num)
	else:
		return "{:.03f}".format(num)
		#return str(round(num, 3))

def format_real(num):
	if int(num) == round(num, 3):
		return str(int(num))
	else:
		return "{:.03f}".format(num)
		#return str(round(num, 3))

def format_complex(num):
	if num.imag == 0:
		if num.real == 0:
			return "."
		return format_real(num.real)
	else:
		if num.real == 0:
			return format_real_coeff(num.imag) + "i"
		else:
			return f"{format_real(num.real)}{format_real_coeff(num.real)}i"
		
def format_complex_coeff(num):
	if num.imag == 0:
		return format_real_coeff(num.real)
	else:
		if num.real == 0:
			return format_real_coeff(num.imag) + "i"
		else:
			if num.real > 0:
				return f"{format_real(num.real)}+{format_real_coeff(num.real)}i"
			else:
				return f"{format_real(num.real)}{format_real_coeff(num.real)}i"

def print_matrix(matrix, labels_as_qbits=False):
	rows = len(matrix)
	cols = len(matrix[0])
	max_len = 0
	qbits = len(bin(len(matrix) - 1)) - 2
	for row in matrix:
		for val in row:
			max_len = max(max_len, len(format_complex(val)))
	max_len += 1
	if labels_as_qbits:
		header_len = qbits + 2
		max_len = max(max_len, qbits+1)
	else:
		header_len = len(str(rows+1)) + 2
	print(" "*header_len, end="")
	for i in range(cols):
		if labels_as_qbits:
			label = "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:]
		else:
			label = str(i+1)
		print(label.ljust(max_len), end="")
	print()
	for i, row in enumerate(matrix):
		if labels_as_qbits:
			label = "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:]
		else:
			label = str(i+1)
		print((label + ":").ljust(header_len), end="")
		for j, val in enumerate(row):
			formatted = format_complex(val)
			if j != 0:
				print(" "*(max_len-len(formatted)), end="")
			print(formatted, end="")
		print()


def is_hermitian(matrix, atol=1e-8, rtol=1e-5):
	"""
	Checks if a square 2D array is Hermitian.

	Args:
		matrix (np.ndarray): The input square 2D array.
		atol (float, optional): Absolute tolerance for comparison. Defaults to 1e-8.
		rtol (float, optional): Relative tolerance for comparison. Defaults to 1e-5.

	Returns:
		bool: True if the matrix is Hermitian, False otherwise.
	"""
	if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
		raise ValueError("Input must be a square 2D array.")

	# Calculate the complex conjugate transpose (Hermitian adjoint)
	conjugate_transpose = matrix.conj().T

	# Compare the matrix with its conjugate transpose using a tolerance
	return np.allclose(matrix, conjugate_transpose, atol=atol, rtol=rtol)


def characteristic_polynomial(matrix) -> Polynomial:
	# create characteristic polynomial
	n = matrix.shape[0]
	assert matrix.shape == (n, n)

	new_matrix = []
	for i in range(n):
		new_row = []
		for j in range(n):
			if i == j:
				new_row.append(Polynomial([ -matrix[i, j], 1 ]))
			else:
				new_row.append(matrix[i, j])
		new_matrix.append(new_row)

	new_matrix = np.array(new_matrix)
	
	k = math.ceil(math.log2(n))
	print(k)

	matrices = [None] * 2**k

	matrices[0] = matrix

	for i in range(1, 2**k):
		tmp = copy.deepcopy(matrices[i - 1])
		matrices[i] = tmp @ matrix
	
	e = np.array([1]*16)

	vectors = [e]

	for i in range(n - 1):
		vectors.append(matrices[i] @ e)

	U = np.column_stack(vectors)

	print_matrix(U)

	return Polynomial([0])
