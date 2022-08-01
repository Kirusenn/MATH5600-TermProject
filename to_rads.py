import numpy as np


def to_rads(pi, dms):
	theta = dms[0] + (dms[1] / 60) + (dms[2] / 3600)
	theta = pi * theta / 180

	return theta
