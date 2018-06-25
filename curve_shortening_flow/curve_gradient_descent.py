import numpy as np
import matplotlib.pyplot as plt 
import time

# curve = np.array([[0,0],[1,1],[1,2],[0,3],[-1,2],[-1,1]])
alpha = 0.001
eps = 0.07
MIN_DIST = 0.05

def rotate_vector(vector, angle):
	rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)], 
								[np.sin(angle), np.cos(angle)]])
	return rotation_matrix.dot(vector.transpose())

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    return (np.arctan2(*v2[::-1]) - np.arctan2(*v1[::-1])) % (2 * np.pi) 

def inward_normal(pt_index, curve):
	n = len(curve)
	pt = curve[pt_index]
	prev_pt = curve[((pt_index - 1) % n)]
	next_pt = curve[((pt_index + 1) % n)]
	v1 = unit_vector(prev_pt - pt)
	v2 = unit_vector(next_pt - pt)
	return unit_vector((v1 + v2))


def curvature(pt_index, curve):
	n = len(curve)
	pt = curve[pt_index]
	prev_pt = curve[((pt_index - 1) % n)]
	next_pt = curve[((pt_index + 1) % n)]
	v1 = pt - prev_pt
	v2 = next_pt - pt
	return 2 * np.sin(angle_between(v1, v2) / 2.0) / np.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2))

def point_flow(pt_index, curve):
	n = len(curve)
	pt = curve[pt_index]
	prev_pt = curve[((pt_index - 1) % n)]
	next_pt = curve[((pt_index + 1) % n)]
	v1 = prev_pt - pt 
	v2 = next_pt - pt
	# return alpha * (unit_vector(v1) + unit_vector(v2)) / np.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2)) 
	return alpha * 2 * (unit_vector(v1) + unit_vector(v2)) / (np.linalg.norm(v2) + np.linalg.norm(v2))
	# return alpha * (unit_vector(v1) + unit_vector(v2))
	# return alpha * curvature(pt_index, curve) * inward_normal(pt_index, curve)

def preprocess(curve):
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
	curve = preprocess(curve)
	return np.array([pt + point_flow(pt_index, curve) for pt_index, pt in enumerate(curve)])
	
def flow(curve, steps):
	print "Flowing the curve: {}".format(curve)
	# plt.plot(curve[:,0],curve[:,1], 'ro')
	# plt.show()
	# pt_pos = [curve[2]]
	for i in range(steps):
		curve = flow_step(curve)
		# pt_pos.append(curve[2])
		# plt.plot(curve[:,0],curve[:,1], 'ro')
		# plt.show()
	# pt_pos = np.array(pt_pos)
	# plt.plot(pt_pos[:,0],pt_pos[:,1], 'ro')
	# plt.show()
	# print "Final curve: {}".format(curve)
	print "Finished flowing"

# curve = np.array([[np.cos(theta), 2*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 50, endpoint=False)])
# start = time.time()
# flow(curve, 2000)
# print "Time to flow: {}".format(time.time() - start)