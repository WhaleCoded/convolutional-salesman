import time
import numpy as np
import copy
import heapq
import itertools
from typing import List, Tuple
import math
from scenario_gen.scenario_classes import TSPSolution, City, State
from queue import PriorityQueue as PQ



class TSPSolver:
	def __init__( self):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		self._scenario.makeDistMatrix()
		matrix = self._scenario.getMatrix()

		route = [0] * len(matrix)
		visited = self._scenario.makeVisitedMap()
		cities = self._scenario.getCities()
		ncities = len(cities)
		sum = 0
		foundTour = False
		i = 0
		j = 0
		counter = 1
		min = np.inf
		count = 1
		bssf = None
		start_time = time.time()

		#Mark beginning city as visited and add it to route list
		visited[cities[0]._name] = True
		route[0] = 0

		#Loop through while the num of cities is less than ncities
		# at worst n^2 times
		while i < ncities and j < ncities:

			if counter >= len(matrix[i]) - 1:
				break

			#Get the min of the current row, only considering unvisited nodes
			if not visited[cities[j]._name]:
				#this could be the problem, add equals sign? 40 points Seed: 23 fails
				if matrix[i][j] <= min:
					min = matrix[i][j]
					route[counter] = j 

			#if its the last cell in a row, add up the sum and add the next city to the
			#routes and mark it as visited
			if j == len(matrix) - 1:
				sum += min
				min = np.inf
				j = 0
				i = route[counter]
				visited[cities[i]._name] = True
				counter += 1
			else:
				j += 1



		#run it one more time for to set up the last city
		# runs at most n times
		j = 0
		i = route[len(matrix) - 2]
		min = np.inf
		while j < ncities:
			if (not visited[cities[j]._name]) and matrix[i][j] <= min:
				min = matrix[i][j]
				route[len(matrix) - 1] = j
			j += 1
		for i in range(ncities):
			route[i] = cities[route[i]]

		#set up the bssf and foundTour
		bssf = TSPSolution(route)
		foundTour = True

		end_time = time.time()
		results = {}
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''
	#Time: O(n! * work for each node)
	#Space: O()
	def branchAndBound( self, time_allowance=60.0 ):
	#Use function to get the initial BSSF solution/cost
		currentSolution = self.getInitialBSSF()['soln']
		bssf = currentSolution.cost

		#set up stats to report
		numPruned = 0
		queueMaxSize = 1
		numSolutionsFound = 0
		numStatesCreated = 1

		#set up all other variables
		self._scenario.makeDistMatrix()
		matrix = self._scenario.getMatrix()
		route = []
		visitedRow = self._scenario.makeVisitedMap()
		visitedCol = self._scenario.makeVisitedMap()
		cities = self._scenario.getCities()
		ncities = len(cities)
		#sum = 0
		queueEmpty = False
		self.queue = PQ()

		#always starts with first city, so append to route list
		route.append(0)

		#reduce the initial matrix to get the first lower bound 
		# and reduced cost matrix
		# O(n^2) to reduce the initial matrix
		matrix, bound = self.reduceInitialMatrix(matrix, visitedRow, 0)

		#create the inital state with the data from reducing the initial matrix
		initialState = State(matrix, bound, route, (0, 0), visitedRow, visitedCol)

		#add the state to be the first one on the queue 
		# so it is expanded as soon as the while loop starts
		# O(logn) to put on the priority queue
		self.queue.put((initialState.getBound(), initialState)) #add the intital state to the PQ

		#make sure all variables are updated
		matrix = initialState.getMatrix()
		route = initialState.getRoute()
		bound = initialState.getBound()
		visitedRow = initialState.getVisitedRow()
		visitedCol = initialState.getVisitedCol()

		startTime = time.time() #start clock
		# while loop can loop at most n! times
		while self.queue.qsize() > 0 and time.time() - startTime < time_allowance: 
			#pop the top state off of the priority queue
			# O(logn) to pop off of queue
			currentState = self.queue.get()[1]

			#prune the popped off state while its cost
			# is higher than the current bssf cost
			while currentState.bound > bssf:
				numPruned += 1
				
				#if this is the last queue then set boolean
				if self.queue.qsize() == 0:
					queueEmpty = True
					break
				currentState = self.queue.get()[1]

			#check boolean to see if while loop should end
			if queueEmpty: 
				break

				
			#expand the current state
			i = currentState.getPath()[1]

			#iterate through each cell in the current states matrix
			#if it leads to an unvisited node and the cost is less than the bssf cost,
			# create a state for that path, reduce it, and evaluate
			for j in range(len(currentState.getMatrix()[i])):
				if currentState.getMatrix()[i][j] < bssf and not currentState.getVisitedCol()[cities[j].getName()] and not currentState.getVisitedRow()[cities[j].getName()]:
					tempState = State(currentState.getMatrix(), currentState.getBound(), currentState.getRoute(), (i, j), currentState.getVisitedRow(), currentState.getVisitedCol())
					tempState.addCityToRoute(j)
					numStatesCreated += 1

					#if reduce state function returns false, then
					# the current state can be pruned
					# O(n^2) function call
					if not self.reduceStateMatrix(tempState, bssf):
						numPruned += 1
						
					#else check states cost against the bssf cost
					# and evaluate
					else:
						if tempState.getBound() < bssf:
							tempBound = self.getTempBound(tempState)
							self.queue.put((tempBound, tempState))
							if self.queue.qsize() > queueMaxSize:
								queueMaxSize = self.queue.qsize()
						else:
							numPruned += 1

			#this means that a leaf node has been reached
			# and a solution found
			if len(currentState.getRoute()) == ncities: 
				numSolutionsFound += 1

				if(currentState.getBound() < bssf):

					#routes are stored as city indexes to save time
					#change solutions route to store the city objects
					routeCities = []
					for i in range(ncities):
						routeCities.append(cities[currentState.getRoute()[i]])

					#update current soltuion and best cost
					currentSolution = TSPSolution(routeCities)
					bssf = currentSolution.cost

		#if nodes left on queue when terminated, add them to the number pruned
		while self.queue.qsize() > 0:
			current = self.queue.get()
			if current[1].getBound() > bssf:
				numPruned += 1

		#stop timer
		endTime = time.time()

		#create and return result map
		results = {}
		results['cost'] = bssf 
		results['time'] = endTime - startTime
		results['count'] = numSolutionsFound
		results['soln'] = currentSolution
		results['max'] = queueMaxSize
		results['total'] = numStatesCreated
		results['pruned'] = numPruned
		return results

	#Time: O(n^2)
	#Space: O(n^2)
	def reduceStateMatrix(self, state, bssf):
		#set up variables
		cities = self._scenario._cities
		matrix = state.getMatrix()
		bound = state.getBound()
		visitedRow = state.getVisitedRow()
		visitedCol = state.getVisitedCol()
		path = state.getPath()

		#add the starting cells value to the bound
		bound += matrix[path[0]][path[1]]
		#if it causes the bound to be greater than the bssf cost, return
		# false to prune the state
		if bound > bssf:
			return False

		#update which rows and columns should be skipped
		visitedRow[cities[path[0]].getName()] = True
		visitedCol[cities[path[1]].getName()] = True

		#set the row and column in the states matrix to infinity
		# O(n) function call
		self.setToInf(matrix, path)

		#look through the rows 
		# O(n^2), loops through at worst each cell
		for i in range(len(matrix)): 
			min = np.inf
			#if the row is unvisited
			if not visitedRow[cities[i].getName()] and not visitedCol[cities[i].getName()]:
				#look through each cell in the row
				for j in range(len(matrix[i])):
					#if the cells column is unvisited then check value
					if not visitedCol[cities[j].getName()]:
						#if value is less than current min, update current min
						if matrix[i][j] < min:
							min = matrix[i][j]

				#update values for min less than infinity
				if min != np.inf:	
					#subtract the min from each value
					for j in range(len(matrix[i])):
						matrix[i][j] -= min

				#add min to the bound
				bound += min

				#return immediately to prune if bound exceeds bssf cost
				if bound > bssf:
					return False
		
		#look through columns
		# O(n^2), loops through at worst each cell
		for i in range(len(matrix)):
			#check if the column is unvisited
			if not visitedCol[cities[i].getName()]:
				min = np.inf
				#check if the cells row is unvisited
				for j in range(len(matrix[i])):
					if not visitedRow[cities[j].getName()]:
						#check value against current min and update if needed
						if matrix[j][i] < min: 
							min = matrix[j][i]
				#subtract the min from each value
				if min != np.inf:	
					for j in range(len(matrix[j])):
						matrix[j][i] -= min
			#add min to bound
			bound += min
			#return immediately to prune if bound exceeds bssf cost
			if bound > bssf:
				return False
		
		#make sure variables get updated
		state.bound = bound
		state.visitedRow = visitedRow
		state.visitedCol = visitedCol

		#return to continue to evaluate the state
		return True

	#Time: O(n)
	#Space: O(n^2)
	def setToInf(self, matrix, path):
		for j in range(len(matrix)):
			matrix[path[0]][j] = np.inf
			matrix[j][path[1]] = np.inf

	#Time: O(n^2)
	#Space: O(n^2)
	def reduceInitialMatrix(self, matrix, visited, bound):
		#set up variables
		cities = self._scenario.getCities()
		visited = {}
	
		#initialize map value for each city to unvisited
		# O(n) loop
		for i in range(len(matrix)):
			visited[cities[i].getName()] = False
		
		#look at each cell and get min for each unvisited citys row
		# O(n^2) loop
		for i in range(len(matrix)): 
			min = np.inf
			#if not visited[cities[i].getName()]:
			for j in range(len(matrix[i])):
				
				#if not visited[cities[j].getName()]:
					if matrix[i][j] < min:
						min = matrix[i][j]
					#if its infinity, then prune
					#if not, then add the min to bound and subtract it from all values
			if min != np.inf:	
				#subtract the min from each value
				for j in range(len(matrix[i])):
					matrix[i][j] -= min
			#add min to the bound
			bound += min
		
		#look through columns and get min for each unvisited citys row
		# O(n^2) loop
		for i in range(len(matrix)):
			min = np.inf
			#only check unvisited nodes
			if not visited[cities[j].getName()]:
				for j in range(len(matrix[j])):
					if matrix[j][i] < min: 
						min = matrix[j][i]
				if min != np.inf:	
					#subtract the min from each value
					for j in range(len(matrix[j])):
						matrix[j][i] -= min
				bound += min

		#return the reduced matrix and bound value		
		return matrix, bound

				

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass