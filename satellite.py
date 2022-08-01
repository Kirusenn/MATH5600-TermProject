import numpy as np
import sys
import math

from numpy import sin, cos
from to_rads import to_rads

log = open("satellite.log", "w")

data = np.loadtxt('data.dat', usecols=0)

pi = data[0]
c = data[1]
R = data[2]
s = data[3]

out_string = ''

satellites = np.zeros((9, 24))
for sat in range(0, 24):
	satellites[:, sat] = data[sat * 9 + 4:sat * 9 + 13]

# input_coords = np.array([[0], [40], [45], [55.0], [1], [111], [50], [58.0], [-1], [1372.00]])
# input_coords = np.array([[(0*s/8)], [0], [0], [0.0], [1], [0], [0], [0.0], [1], [0.00]])


#   Method to get coordinates of a satellite at a given time
def get_sat_coords(sat, time):
	return (R + satellites[7, sat]) * (np.multiply(satellites[0:3, sat], float(cos(2 * pi * time / satellites[6, sat] + satellites[8, sat]))) + np.multiply(satellites[3:6, sat], float(sin(2 * pi * time / satellites[6, sat] + satellites[8, sat]))))


# def main(argv):
# 	out_string = ''

log.write('Received:\n')

for line in sys.stdin:
	log.write(line)
	input_coords = line.split()
	t_v = float(input_coords[0])
	psi = np.array(input_coords[1:4], dtype='float')
	ns = int(input_coords[4])
	lamb = np.array(input_coords[5:8], dtype='float')
	ew = int(input_coords[8])
	h = float(input_coords[9])

	d = R + h
	t_rads = 2 * pi * t_v / s
	rot_mat = np.array([[float(cos(t_rads)), -float(sin(t_rads)), 0], [float(sin(t_rads)), float(cos(t_rads)), 0], [0, 0, 1]])

	psi_rads = to_rads(pi, psi)
	lamb_rads = to_rads(pi, lamb)

	p_z = d * sin(psi_rads)
	p_xy = d * cos(psi_rads)

	cart_coords = np.array([[p_xy * cos(lamb_rads)], [p_xy * sin(lamb_rads) * ew], [p_z * ns]])
	cart_coords = cart_coords.reshape(cart_coords.shape[0:2])
	cart_coords = np.matmul(rot_mat, cart_coords)

	# set initial satellite coordinates to their positions at t_v
	sat_coords = np.zeros((3, 24))
	for sat in range(0, 24):
		sat_coords[:, sat] = get_sat_coords(sat, t_v)

	final_sats = np.zeros((5, 0))	# will hold sat number, t_s, and coordinates of satellites above horizon
	xTx = np.dot(cart_coords.transpose(), cart_coords)
	t_s_tol = 0.01 / c

	for sat in range(0, 24):
		# if np.dot(cart_coords.transpose(), sat_coords[:, sat]) > xTx:	# determine if satellite is above horizon; if so, add sat to final_sats
		d_vs = np.array(sat_coords[:, sat].reshape(3, 1) - cart_coords)	# distance between satellite and vehicle
		t_k = float(t_v - np.dot(d_vs.transpose(), d_vs) / c)	# t_0
		t_diff = float('inf')

		while t_diff > t_s_tol: # calculate t_k1
			f_tk = np.dot(d_vs.transpose(), d_vs) - math.pow(c, 2)*math.pow((t_v - t_k), 2)

			f_prime_tk = float((4*pi*(R+satellites[7, sat])/satellites[6, sat]) * np.dot(d_vs.transpose(), (-np.multiply(satellites[0:3, sat], float(sin(2*pi*t_k/satellites[6, sat] + satellites[8, sat])))) + np.multiply(satellites[3:6, sat], float(cos(2*pi*t_k/satellites[6, sat] + satellites[8, sat])))) + 2*math.pow(c, 2)*(t_v-t_k))

			t_k1 = float(t_k - (f_tk / f_prime_tk))

			t_diff = float(abs(t_k1 - t_k))

			t_k = t_k1

			sat_coords[:, sat] = get_sat_coords(sat, t_k)	# update satellite coordinates to time t_k
			d_vs = sat_coords[:, sat].reshape(3, 1) - cart_coords	# update distance

		if np.dot(cart_coords.transpose(), sat_coords[:, sat]) > xTx:	# double check satellite is above horizon at final time t_k
			final_sats = np.append(final_sats, [[sat], [t_k], [sat_coords[0, sat]], [sat_coords[1, sat]], [sat_coords[2, sat]]], 1)

	for sat in range(0, final_sats.shape[1]):
		out_string += f'{math.floor(final_sats[0, sat])} {final_sats[1, sat]:.15E} {final_sats[2, sat]:.15E} {final_sats[3, sat]:.15E} {final_sats[4, sat]:.15E}\n'

log.write('\nOutput:\n')
log.write(out_string[:-1])
log.close()
sys.stderr.close()
print(out_string[:-1])
exit()


# if __name__ == "__main__":
# 	main(sys.argv[1:])
