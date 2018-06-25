import numpy as np
import matplotlib.pyplot as plt 
import time

alpha = 0.001
eps = 0.07
MIN_DIST = 0.05

def rotate_vector(vector, angle):
	""" 
	Rotates the input vector by the given angle counterclockwise. 
	"""
	rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)], 
								[np.sin(angle), np.cos(angle)]])
	return rotation_matrix.dot(vector.transpose())

def unit_vector(vector):
    """ 
    Returns the unit vector of the vector.  
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" 
	Returns the counterclockwise angle from vector v1 to vector v2. 
	Vectors should be numpy arrays of shape (2,1).
	"""
    return (np.arctan2(*v2[::-1]) - np.arctan2(*v1[::-1])) % (2 * np.pi) 

def inward_normal(pt_index, curve):
	""" 
	Returns the inward pointing normal vector to the curve at the point 
	specified by pt_index.
	NOTE: unused by point_flow in current implementation (final equation 
		for flow has terms cancel out between normal and curvature.)

	Arguments: 
	pt_index -- the index of the desired point in curve. 
	curve -- a numpy array of shape (length, 2). 
	"""
	n = len(curve)
	pt = curve[pt_index]
	prev_pt = curve[((pt_index - 1) % n)]
	next_pt = curve[((pt_index + 1) % n)]
	v1 = unit_vector(prev_pt - pt)
	v2 = unit_vector(next_pt - pt)
	return unit_vector((v1 + v2))


def curvature(pt_index, curve):
	"""
	Computes a discretized version of the curvature of the input curve at the 
	specified point. 
	NOTE: unused by point_flow in current implementation (final equation 
		for flow has terms cancel out between normal and curvature.)

	Arguments:
	pt_index -- the index of the desired point in curve. 
	curve -- a numpy array of shape (length, 2). 
	"""
	n = len(curve)
	pt = curve[pt_index]
	prev_pt = curve[((pt_index - 1) % n)]
	next_pt = curve[((pt_index + 1) % n)]
	v1 = pt - prev_pt
	v2 = next_pt - pt
	return 2 * np.sin(angle_between(v1, v2) / 2.0) / np.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2))

def point_flow(pt_index, curve):
	"""
	Computes the displacement by which to move a point on the input curve at a step 
	of the discretized curve-shortening flow process. 

	Arguments:
	pt_index -- the index of the desired point in curve. 
	curve -- a numpy array of shape (length, 2). 
	"""
	n = len(curve)
	pt = curve[pt_index]
	prev_pt = curve[((pt_index - 1) % n)]
	next_pt = curve[((pt_index + 1) % n)]
	v1 = prev_pt - pt 
	v2 = next_pt - pt
	return alpha * 2 * (unit_vector(v1) + unit_vector(v2)) / (np.linalg.norm(v2) + np.linalg.norm(v2))

def preprocess(curve):
	"""
	Performs some cleaning of the input discretized curve for flow stability. Presently 
	just removes points that are too close to each other. 
	"""
	new_curve = np.empty((0,2), float)
	n = len(curve)
	last_pt = curve[-1]
	for pt_index, pt in enumerate(curve):
		dist = np.linalg.norm(pt - last_pt)
		if dist > MIN_DIST: 
			new_curve = np.append(new_curve, [pt], axis=0)
			last_pt = pt 
	return new_curve

def flow_step(curve):
	"""
	Performs a step of the discretized curve-shortening flow for the input curve. 
	"""
	curve = preprocess(curve)
	return np.array([pt + point_flow(pt_index, curve) for pt_index, pt in enumerate(curve)])
	
def flow(curve, steps):
	"""
	Performs the discretized curve-shortening flow on the input curve for the given 
	number of steps.
	"""
	print "Flowing the curve: {}".format(curve)
	for i in range(steps):
		curve = flow_step(curve)
	print "Finished flowing"
