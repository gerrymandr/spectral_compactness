import numpy as np 
import shapely.geometry as geom
import geopandas as gpd

def check_closed_curve(curve):
	return (np.array_equal(curve[0], curve[-1]))

def close_curve(curve):
	if check_closed_curve(curve):
		return curve
	else:
		return np.append(curve, np.array([curve[0]]), axis=0)

def subdivide_curve(curve, threshold=0.0):
	pts = None
	new_curve = np.empty((0,2), float)
	is_closed = check_closed_curve(curve)
	if is_closed:
		pts = curve[:-1]
	else:
		pts = curve
	n = len(pts)
	for pt_index, pt in enumerate(pts):
		new_curve = np.append(new_curve, np.array([pt]), axis=0)
		next_pt = curve[((pt_index + 1) % n)]
		if np.linalg.norm(next_pt - pt) < threshold:
			continue
		mid_pt = pt + (next_pt - pt) / 2.0 
		new_curve = np.append(new_curve, np.array([mid_pt]), axis=0)
	if is_closed:
		return close_curve(new_curve)
	else:
		return new_curve

def normalize_curve(curve, norm):
	region = geom.Polygon(close_curve(curve))
	scale_factor = np.sqrt(norm / region.area)
	return np.array([scale_factor * pt for pt in curve])

def curve_from_shapefile(filename, tolerance=0): 
	"""
	Reads in the shapefile at filename specifying a single polygonal outline
	and returns a simplified polygon (default tolerance=0 doesn't simplify read shape).
	"""
	shape_data = gpd.read_file(filename)
	curve_outline = shape_data['geometry'][0]
	return curve_outline.simplify(tolerance)