from scenario_gen.solver import TSPSolver
import random
from scenario_gen.scenario_classes import Scenario, City
from typing import List
import numpy as np

class TSPMaster():

  def __init__( self, max_seed: int = 10000, max_time: int = 10, num_cities: int = 1000 ):
    self._MAX_SEED = max_seed
    self._scenario = None
    self.solver = TSPSolver()
    self.diff = 'Easy'
    self.size = num_cities
    SCALE = 1.0
    self.max_time = max_time
    self.data_range		= { 'x':[-1.5*SCALE,1.5*SCALE], \
                'y':[-SCALE,SCALE] }
    
  def get_new_seed(self):
    self.seed = random.randint(0, self._MAX_SEED-1)

  def set_size(self, new_size: int):
    self.size = new_size

  def get_size(self):
    return self.size


  def newPoints(self):
    # TODO - ERROR CHECKING!!!!

    ptlist = []
    RANGE = self.data_range
    xr = self.data_range['x']
    yr = self.data_range['y']
    npoints = int(self.size)
    while len(ptlist) < npoints:
      x = random.uniform(0.0,1.0)
      y = random.uniform(0.0,1.0)
      if True:
        xval = xr[0] + (xr[1]-xr[0])*x
        yval = yr[0] + (yr[1]-yr[0])*y
        ptlist.append( (xval,yval) )
    return ptlist

  def generateNetwork(self):
    points = self.newPoints() # uses current rand seed
    diff = self.diff
    rand_seed = self.get_new_seed
    self._scenario = Scenario( city_locations=points, difficulty=diff, rand_seed=rand_seed )
    return self.create_cost_matrix(self._scenario.getCities())

  def create_cost_matrix(self, cities: List[City]) -> np.ndarray:

    #O(n^2)
    cost_matrix = np.zeros((len(cities), len(cities)), float)
    for i in range(len(cities)):
      for x in range(len(cities)):
        cost_matrix[i][x] = cities[i].costTo(cities[x])

    return cost_matrix

  def generate(self):
     return self.generateNetwork()

  def solve(self):								
    self.solver.setupWithScenario(self._scenario)
    results = self.solver.greedy(self.max_time)

    return results['soln'].enumerateEdges()