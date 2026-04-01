from qwalk import *

# === build the Hamiltonian ===

QBITS = 4

def ga(m, p):
	return get_gate_at_pos(m, p, QBITS)

X0 = ga(X, 0)
Y0 = ga(Y, 0)
Z0 = ga(Z, 0)

# sigma plus
sp = 0.5*( X + complex(0, 1)*Y )
sp1 = ga(sp, 1)
sp2 = ga(sp, 2)
sp3 = ga(sp, 3)
# sigma minus
sm = 0.5*( X - complex(0, 1)*Y )
sm1 = ga(sm, 1)
sm2 = ga(sm, 2)
sm3 = ga(sm, 3)


H1 = Z0@(sp1@sm2 + sp2@sm1) \
   + X0@(sp2@sm3 + sp3@sm2) \
   + Y0@(sp1@sm3 + sp3@sm1)

H2 = Z0@(sm1@sm2 + sp2@sp1) \
   + X0@(sm2@sm3 + sp3@sp2) \
   + Y0@(sm3@sm1 + sp1@sp3)

print_matrix(H1)
print()
print_matrix(H2)

a = 1
b = 1
H = a*H1 + b*H2

print_matrix(H)


# === Build the initial state and target states ===

# build initial state with the 4 states in the pyramid
def build_state_vector(s0000_prob, s1101_prob, s1011_prob, s0110_prob):
   assert s0000_prob + s1101_prob + s1011_prob + s0110_prob - 1 < 0.00001
   initial = (s0000_prob**0.5)*krons(ket0, ket0, ket0, ket0) + (s1101_prob**0.5)*krons(ket1, ket1, ket0, ket1) + (s1011_prob**0.5)*krons(ket1, ket0, ket1, ket1) + (s0110_prob**0.5)*krons(ket0, ket1, ket1, ket0)
   return initial / np.linalg.norm(initial)

# initial state with uniform mixing
initial = build_state_vector(0.25, 0.25, 0.25, 0.25)

# target1: 50/50 mix between two states
target1 = build_state_vector(0.5, 0, 0, 0.5)
# target2: uniform mixing
target2 = build_state_vector(0.25, 0.25, 0.25, 0.25)
# target3: 1/6 on two states, 1/3 on other two states
target3 = build_state_vector(1/6, 1/3, 1/3, 1/6)


# === Simulate the quantum walk ===

simulate_qwalk2(H, initial, [target1, target2, target3], time_limit=10, time_increment=0.0001)
