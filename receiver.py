import numpy as np
import sys
import math

from numpy import sin, cos
from to_rads import to_rads
from to_dms import to_dms

log = open("receiver.log", "w")

data = np.loadtxt('data.dat', usecols=0)

pi = data[0]
c = data[1]
R = data[2]
s = data[3]

psi_rads = to_rads(pi, np.array([[40], [45], [55.0]]))
lamb_rads = to_rads(pi, np.array([[111], [50], [58.0]]))

p_z = (R+1372.00) * sin(psi_rads)
p_xy = (R+1372.00) * cos(psi_rads)

# input = '3 1.212291727353893E+04 2.605234313778725E+07 2.986153965269792E+06 4.264669833325115E+06\n4 1.212291811597410E+04 -1.718355633086311E+07 -1.864083427618644E+07 7.941901319733662E+06\n8 1.212291517247339E+04 1.849827925661685E+07 -1.417239006438451E+07 -1.275876685529343E+07\n11 1.212292947400401E+04 -2.903225428514331E+06 -1.966135853780249E+07 1.763041037014707E+07\n14 1.212293081680465E+04 1.477645012869009E+06 -1.521487230846215E+07 2.172908956016601E+07\n15 1.212291559232703E+04 2.652632365283036E+07 8.475085779779141E+05 -1.210367686336006E+06\n17 1.212293212673538E+04 4.939777113795485E+06 -1.796566328317718E+07 1.893839095916287E+07\n20 1.212293029017580E+04 1.790346111594521E+07 -1.680512822049417E+07 1.014311849596406E+07\n3 1.212391727495149E+04 2.605158357529224E+07 2.988333259818807E+06 4.267782188491394E+06\n4 1.212391811479538E+04 -1.718247120929847E+07 -1.864309319017043E+07 7.938946196116364E+06\n8 1.212391517004298E+04 1.849870247761143E+07 -1.416952246684813E+07 -1.276133797714327E+07\n11 1.212392947448364E+04 -2.900527563545101E+06 -1.966342525792576E+07 1.762854940826564E+07\n14 1.212393081770695E+04 1.481513660682393E+06 -1.521474853899824E+07 2.172891279905284E+07\n15 1.212391559038413E+04 2.652610787655970E+07 8.497275371656624E+05 -1.213536688477031E+06\n17 1.212393212691120E+04 4.942411176984275E+06 -1.796329047680592E+07 1.893995444197517E+07\n20 1.212393029104404E+04 1.790426081992824E+07 -1.680258096219933E+07 1.014592651073815E+07'

log.write('Received:\n')
out_string = ''
input = ''
for line in sys.stdin:
	log.write(line)
	input = input + line

input = np.array(input.split(), dtype='float')
input = input.reshape(int(input.shape[0]/5), 5)

t_rads = 2 * pi * input[0, 1] / s
rot_mat = np.array([[float(cos(t_rads)), -float(sin(t_rads)), 0], [float(sin(t_rads)), float(cos(t_rads)), 0], [0, 0, 1]])

cart_coords = np.array([[p_xy * cos(lamb_rads)], [p_xy * sin(lamb_rads) * -1], [p_z]])
cart_coords = cart_coords.reshape(cart_coords.shape[0:2])
cart_coords = np.matmul(rot_mat, cart_coords)

x0 = cart_coords
tol = .01

current_sat = 0
while current_sat < input.shape[0]-1:
	step = float('inf')
	start_sat = current_sat
	current_max = input[current_sat, 0]
	while current_sat < input.shape[0]-1 and input[current_sat+1, 0] > current_max:
		current_sat = current_sat + 1
		current_max = input[current_sat, 0]

	sat_info = np.array(input[start_sat:current_sat+1, :])

	x_k = x0

	while step > tol:
		F = np.zeros((current_sat, 1))
		J = np.zeros((current_sat, 3))
		for sat in range(0, sat_info.shape[0]-1):
			d_1 = np.linalg.norm(sat_info[sat, 2:5].reshape(3,1) - x_k, 2)
			d_2 = np.linalg.norm(sat_info[sat+1, 2:5].reshape(3,1) - x_k, 2)

			# Jacobian
			J[sat, 0] = ((sat_info[sat,2] - x_k[0]) / d_1) - ((sat_info[sat+1,2] - x_k[0]) / d_2)
			J[sat, 1] = ((sat_info[sat,3] - x_k[1]) / d_1) - ((sat_info[sat+1,3] - x_k[1]) / d_2)
			J[sat, 2] = ((sat_info[sat,4] - x_k[2]) / d_1) - ((sat_info[sat+1,4] - x_k[2]) / d_2)

			# F
			F[sat, 0] = d_2 - d_1 - c*(sat_info[sat, 1] - sat_info[sat+1, 1])

		J_new = np.matmul(J.transpose(), J)
		F = np.matmul(J.transpose(), F)
		J = J_new

		F = -F

		s_k = np.linalg.solve(J, F)
		x_k = x_k + s_k
		step = float(np.linalg.norm(s_k, np.inf))

	tmp = x_k - sat_info[0, 2:5].reshape(3,1)
	# calculate t_v based on one of the satellites
	t_v = float(np.linalg.norm(tmp, 2)/c + sat_info[0,1])
	t_rads = 2 * pi * t_v / s
	# rotate to time 0 to find lat and long
	rot_mat = np.array([[float(cos(t_rads)), float(sin(t_rads)), 0], [-float(sin(t_rads)), float(cos(t_rads)), 0], [0, 0, 1]])
	x_k = np.matmul(rot_mat, x_k)
	h = math.sqrt(np.dot(x_k.transpose(), x_k)) - R
	if x_k[1] >= 0:
		ew = 1
	else:
		ew = -1

	if x_k[2] >= 0:
		ns = 1
	else:
		ns = -1

	if x_k[0] == x_k[1]:
		psi = pi/2
	else:
		psi = abs(np.arctan(x_k[2]/math.sqrt(x_k[0]**2 + x_k[1]**2)))

	if x_k[0] < 0:
		lamb = pi + np.arctan(x_k[1]/x_k[0])
	else:
		if x_k[1] > 0:
			lamb = np.arctan(x_k[1]/x_k[0])
		else:
			lamb = 2*pi + np.arctan(x_k[1]/x_k[0])

	if lamb > pi:
		lamb = abs(2*pi - lamb)

	psi_dms = to_dms(pi, psi)
	lamb_dms = to_dms(pi, lamb)

	out_string += f'{t_v:.2f} {math.floor(psi_dms[0])} {math.floor(psi_dms[1])} {np.float(psi_dms[2]):.4f} {ns} {math.floor(lamb_dms[0])} {math.floor(lamb_dms[1])} {np.float(lamb_dms[2]):.4f} {ew} {np.float(h):.2f}\n'
	current_sat = current_sat + 1

log.write('\nOutput:\n')
log.write(out_string[:-1])
log.close()
sys.stderr.close()
print(out_string[:-1])
exit()
