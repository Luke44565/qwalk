import numpy as np
import math
from functools import reduce
import time
import scipy
from termcolor import cprint
from utils import *


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

def print_qstate_prob_colors(state, sort=True, max_print=8, end=False):
	qbits = len(bin(len(state) - 1)) - 2
	things = []
	for i in range(len(state)):
		coef = np.round(state[i], 3)
		if coef == 0:
			continue
		prob = state[i] * state[i].conjugate()
		assert prob.imag < 0.001

		qbit_str = "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:]
		things.append([qbit_str, prob.real])
	
	if sort:
		things.sort(key=lambda i: i[1], reverse=True)

	first = True
	for i, thing in enumerate(things):
		if sort and i == max_print:
			break
		if not first:
			print(" + ", end="")
		qbit_str, prob = thing

		intensity = int(255*prob)
		color = (255-intensity, intensity, 0)
		cprint(f"{prob*100:04.1f}=|" + qbit_str + ">", color, end="")
		first = False
	if end:
		print()
	else:
		print("\r", end="", flush=True)

def print_qstate_phase_colors(state, sort=True, max_print=8, end=False):
	qbits = len(bin(len(state) - 1)) - 2
	things = []
	for i in range(len(state)):
		coef = np.round(state[i], 3)
		if coef == 0:
			continue
		prob = state[i] * state[i].conjugate()
		assert prob.imag < 0.001

		qbit_str = "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:]
		things.append([qbit_str, coef])
	
	if sort:
		things.sort(key=lambda i: i[1]*i[1].conjugate(), reverse=True)

	first = True
	for i, thing in enumerate(things):
		if sort and i == max_print:
			break
		if not first:
			print(" + ", end="")
		qbit_str, coef = thing

		prob = coef * coef.conjugate()
		intensity = int(255*prob.real)
		color = (255-intensity, intensity, 0)
		cprint(f"{format_complex_coeff(coef).rjust(13)}|{qbit_str}>", color, end="")
		first = False
	if end:
		print()
	else:
		print("\r", end="", flush=True)

def print_pst(state, t, threshold=0.99):
	qbits = len(bin(len(state) - 1)) - 2
	for i in range(len(state)):
		prob = state[i] * state[i].conjugate()
		prob = prob.real
		if prob > threshold:
			print("PST to |" + "0"*(qbits-len(bin(i)[2:])) + bin(i)[2:] + ">", end="")
			print(" ({:.10f}%) at t={:.3f}     ".format(prob*100,t))

def print_pst_target(state, t, target, error=0.001):
	diff = 0
	for i in range(len(state)):
		diff += abs(state[i]*state[i].conjugate() - target[i]*target[i].conjugate())
	if diff < error:
		print("PST with error {:.10f}% at time t={:.3f}. To target state:  ".format(diff*100, t))
		print_qstate_prob_colors(target)
		#print_qstate_phase_colors(state)
		print()

def test_pst_target(state, t, target, error=0.001):
	diff = 0
	for i in range(len(state)):
		diff += abs(state[i]*state[i].conjugate() - target[i]*target[i].conjugate())
	return diff < error

def adj_matrix_to_hamiltonian(mat):
	vertices = mat.shape[0]
	out = np.zeros(shape=(2**vertices, 2**vertices), dtype=np.complex128)
	for row in range(vertices):
		for col in range(vertices):
			if row <= col:
				val = mat[row][col]
				if val == 0:
					continue
				elif val == 1:
					out += get_gate_at_pos(X, row, vertices) @ get_gate_at_pos(X, col, vertices) + \
						   get_gate_at_pos(Y, row, vertices) @ get_gate_at_pos(Y, col, vertices)
				elif val == 1j:
					out += get_gate_at_pos(X, row, vertices) @ get_gate_at_pos(Y, col, vertices) - \
						   get_gate_at_pos(X, col, vertices) @ get_gate_at_pos(Y, row, vertices)
				elif val == -1j:
					out += get_gate_at_pos(X, col, vertices) @ get_gate_at_pos(Y, row, vertices) - \
						   get_gate_at_pos(X, row, vertices) @ get_gate_at_pos(Y, col, vertices)
				else:
					raise ValueError("Weighted graphs not implemented")
			else: # row > col
				# adj matrix should be hermitian, so this can be ignored
				pass
	return out

# Not working
def extract_subgraph(H, bit_cnt):
    bits_in_H = len(bin(len(H) - 1)[2:])
    assert 2**bits_in_H == len(H)
    output_size = math.comb(bits_in_H, bit_cnt)
    out = []
    for row in range(len(H)):
      row_bit_cnt = bin(row).count("1")
      if row_bit_cnt != bit_cnt:
         continue
      next_row = []
      for col in range(len(H)):
         col_bit_cnt = bin(col).count("1")
         if col_bit_cnt == bit_cnt:
            next_row.append(H[row][col])
      out.append(next_row)
    
    out = np.array(out)
    assert out.shape == (output_size, output_size)
    return out

def krons(*args):  # repeated cross product
	return reduce(np.kron, args)


I = np.identity(2, dtype=int)

X = np.array([[0, 1],
              [1, 0]], dtype=int)

Y = np.array([[0 , -1j],
	          [1j,  0 ]])

Z = np.array([[1,  0],
			  [0, -1]], dtype=int)

CNOT = np.array([[1, 0, 0, 0],
				 [0, 1, 0, 0],
				 [0, 0, 0, 1],
				 [0, 0, 1, 0]], dtype=int)

HADAMARD = 1/(2**0.5) * np.array([[1, 1],
								  [1, -1]])

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])

bell = CNOT @ np.kron(HADAMARD @ ket0, ket0)

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

def simulate_qwalk(H, initial_state):
	assert is_hermitian(H)

	#initial_state = ket1
	print("Initial State:")
	#print(initial_state)
	print_qstate(initial_state)

	print("Quantum Walk:")
	t = 0
	TIME_INC = 0.001
	while True:
		new_state = scipy.linalg.expm(-1j*H*t) @ initial_state

		#print_qstate_prob_colors(new_state, sort=False)
		print_qstate_phase_colors(new_state, sort=False)
		#print(new_state)
		#test_pst(new_state, t)
		
		t += TIME_INC
		time.sleep(0.01)

def simulate_qwalk2(H, initial_state, target_states, interactive=True, print_phase=False, time_increment=0.001, time_limit=30, target_tolerance=0.00001):
	assert is_hermitian(H)

	if interactive:
		print("Initial State:")
		print_qstate_prob_colors(initial_state, end=True)

		for i, target in enumerate(target_states):
			print(f"Target State {i+1}:")
			print_qstate_prob_colors(target, end=True)

		print("Quantum Walk:")

	target_state_status = [(False, None) for _ in target_states]  # (found, time)
	
	t = 0
	while t < time_limit:
		new_state = scipy.linalg.expm(-1j*H*t) @ initial_state

		if interactive:
			if print_phase:
				print_qstate_phase_colors(new_state, sort=False)
			else:
				print_qstate_prob_colors(new_state, sort=False)

		for i, target in enumerate(target_states):
			found = test_pst_target(new_state, t, target, error=target_tolerance)
			if found and not target_state_status[i][0]:
				target_state_status[i] = (True, t)
			if not found and target_state_status[i][0]:
				print(f"Detected target state {i+1} between t={target_state_status[i][1]:.4f} and t={t-time_increment:.4f}. State was: ")
				print_qstate_prob_colors(new_state, end=True)
				target_state_status[i] = (False, None)
		
		t += time_increment

	print()
	