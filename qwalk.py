import numpy as np
import math
from functools import reduce
import random
import time
import scipy
from termcolor import colored, cprint


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
		return str(round(num, 3))

def format_real(num):
	if int(num) == round(num, 3):
		return str(int(num))
	else:
		return str(round(num, 3))

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

# just a helper function to pretty print the state vector
def print_qstate(state):
	qbits = len(bin(len(state) - 1)) - 2
	first = True
	for i in range(len(state)):
		coef = np.round(state[i], 3)
		if coef == 0:
			continue
		if first:
			if coef < 0:
				print("-", end="")
		else:
			if coef < 0:
				print(" - ", end="")
			else:
				print(" + ", end="")

		if coef != 1:
			print(str(abs(coef)), end="")

		print("|" + "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:] + ">", end="")
		first = False
	print()

def print_qstate_prob_colors(state):
	qbits = len(bin(len(state) - 1)) - 2
	first = True
	for i in range(len(state)):
		coef = np.round(state[i], 3)
		if coef == 0:
			continue
		if not first:
			print(" + ", end="")

		prob = state[i] * state[i].conjugate()
		assert prob.imag < 0.001
		intensity = int(255*prob.real)
		color = (255-intensity, intensity, 0)

		cprint("|" + "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:] + ">", color, end="")
		first = False
	print("\r", end="")


def print_matrix(matrix):
	rows = len(matrix)
	cols = len(matrix[0])
	max_len = 0
	for row in matrix:
		for val in row:
			max_len = max(max_len, len(format_complex(val)))
	max_len += 1
	header_len = len(str(rows+1)) + 2
	print(" "*header_len, end="")
	for i in range(cols):
		print(str(i+1).ljust(max_len), end="")
	print()
	for i, row in enumerate(matrix):
		print((str(i+1) + ":").ljust(header_len), end="")
		for j, val in enumerate(row):
			formatted = format_complex(val)
			if j != 0:
				print(" "*(max_len-len(formatted)), end="")
			print(formatted, end="")
		print()


def test_pst(state, t):
	qbits = len(bin(len(state) - 1)) - 2
	for i in range(len(state)):
		prob = state[i] * state[i].conjugate()
		prob = prob.real
		if prob > 0.999999:
			print("PST to |" + "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:] + ">", end="")
			print(" {:.10f}%) at t={:.3f}".format(prob*100,t))


def krons(*args):  # repeated cross product
	return reduce(np.kron, args)


I = np.identity(2, dtype=int)

X = np.array([[0, 1],
              [1, 0]], dtype=int)

Y = np.array([[0 , -1j],
	          [1j,  0 ]])

Z = np.array([[1,  0],
			  [0, -1]], dtype=int)

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])

def get_gate_at_pos(gate, pos, size):
	array = []
	good = False
	for i in range(size):
		if i == pos:
			array.append(gate)
			good = True
		else:
			array.append(I)
	assert good
	return krons(*array)

def commutes(A, B):
	return ((A @ B) == (B @ A)).all()

def ket_bra_conv(vec):
	return np.transpose(vec.conj())

I0 = get_gate_at_pos(I, 0, 4)
X0 = get_gate_at_pos(X, 0, 4)
X1 = get_gate_at_pos(X, 1, 4)
X2 = get_gate_at_pos(X, 2, 4)
X3 = get_gate_at_pos(X, 3, 4)
Y0 = get_gate_at_pos(Y, 0, 4)
Y1 = get_gate_at_pos(Y, 1, 4)
Y2 = get_gate_at_pos(Y, 2, 4)
Y3 = get_gate_at_pos(Y, 3, 4)
Z0 = get_gate_at_pos(Z, 0, 4)
Z1 = get_gate_at_pos(Z, 1, 4)
Z2 = get_gate_at_pos(Z, 2, 4)
Z3 = get_gate_at_pos(Z, 3, 4)

signma_plus = 0.5*( X + complex(0, 1)*Y )
signma_minus = 0.5*( X - complex(0, 1)*Y )
signma_plus1 = get_gate_at_pos(signma_plus, 1, 4)
signma_plus2 = get_gate_at_pos(signma_plus, 2, 4)
signma_plus3 = get_gate_at_pos(signma_plus, 3, 4)
signma_minus1 = get_gate_at_pos(signma_minus, 1, 4)
signma_minus2 = get_gate_at_pos(signma_minus, 2, 4)
signma_minus3 = get_gate_at_pos(signma_minus, 3, 4)

Hxx = (X1@X2 + Y1@Y2) + (X2@X3 + Y2@Y3) + (X1@X3 + Y1@Y3)
#print(Hxx)

H = Z0@(signma_plus1@signma_minus2 + signma_plus2@signma_minus1) \
  + X0@(signma_plus2@signma_minus3 + signma_plus3@signma_minus2) \
  + Y0@(signma_plus1@signma_minus3 + signma_plus3@signma_minus1)

#H = Z + X + Y

print("Hamiltonian:")
print_matrix(H)

initial_state = krons(ket1, ket1, ket0, ket1)
#initial_state = ket1
print("Initial State:")
print(initial_state)
print_qstate(initial_state)

print("Quantum Walk:")
t = 0
TIME_INC = 0.001
while True:
	new_state = scipy.linalg.expm(-1j*H*t) @ initial_state

	print_qstate_prob_colors(new_state)
	test_pst(new_state, t)
	
	t += TIME_INC
	time.sleep(0.001)
