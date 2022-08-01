import numpy as np

from numpy import floor


def to_dms(pi, theta):
	theta = theta * 180 / pi
	degrees = np.floor(theta)
	theta = (theta - degrees) * 60
	minutes = np.floor(theta)
	theta = (theta - minutes) * 60
	seconds = np.round(theta, 4)

	return np.array([degrees, minutes, seconds])
