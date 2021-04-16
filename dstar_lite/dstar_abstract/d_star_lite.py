# extends the idea of having a general D* lite solution
# this can be extended to maze-solving, path-finding, propositional satisfiability etc
import copy

class Problem(object):
	def heuristic(self, point, goal):
		return 0
	
	def neighbor_nodes(self, point):
		return []
	
	def distance_between_neighbors(self, point, point2):
		return 1

	def is_goal(self, point, goal):
		return point == goal

	def on_open(self, point, f, g, h):
		pass
	
	def on_close(self, point):
		pass

	def on_update(self, point, f, g, h):
		pass


class PathNotFound(Exception):
	pass

def find_path(problem, start, goal):
	'''
	Finds a path from point start to point goal using the D* lite algorithm.
	'''
	open_set = set()

	open_queue = list() # for f-score

	closed_set = set() # explored

	came_from = dict() # for path

	g_score = dict() # cost to get here

	h_score = dict() # heuristic scores, can change

	p_queue=list() # for keys
	
	# f can be computed rather than stored.
	def f_score(point):
		return g_score[point] + h_score[point]

	def calc_key(point):
		pass
	# we can kick off the algorithm by placing only the start point in the open set.
	# t = copy.deepcopy(start)
	# start = goal
	# goal = t
	g_score[start] = 0
	h = problem.heuristic(start, goal)
	h_score[start] = h
	open_set.add(start)
	open_queue.append( (f_score(start), start) )
	problem.on_open(start, h, 0, h)

	# keep searching until we find the goal, or until all possible pathes have been exhausted
	while open_set:
		open_queue.sort()
		next_f, point = open_queue.pop(0)
		open_set.remove(point)

		if problem.is_goal(point, goal):
			# reached goal, get path
			path = [ point ]
			while point != start:
				point = came_from[point]
				path.append(point)
			path.reverse()
			return path, closed_set

		closed_set.add(point)
		problem.on_close(point)

		for neighbor in problem.neighbor_nodes(point):
			if not neighbor in closed_set:
				tentative_g_score = g_score[point] + problem.distance_between_neighbors(neighbor, point)

				if neighbor not in open_set:
					# new place
					came_from[neighbor] = point
					g = tentative_g_score
					h = problem.heuristic(neighbor, goal)
					g_score[neighbor] = g
					h_score[neighbor] = h
					open_set.add(neighbor)
					f = g + h
					open_queue.append( (f, neighbor) )
					problem.on_open(neighbor, f, g, h)

				else:
					if tentative_g_score < g_score[neighbor]:
						# but we found a better route than before!
						came_from[neighbor] = point
						g = tentative_g_score
						g_score[neighbor] = g
						h = problem.heuristic(neighbor, goal)
						h_score[neighbor] = h
						f = g + h
						
						problem.on_update(neighbor, f, g, h)

	raise PathNotFound("no path from %s to %s." % (str(start), str(goal)))