from __future__ import division
from polynomial import *
import numpy as np

class integration(object):
	def __init__(self, order=8, kind='lobatto'):
		self.order=order
		if kind == 'lobatto':
			self.nodes = lobatto_polynomial(self.order).lobatto_points()
			self.weights = 2/((self.order)*(self.order-1))*np.ones(self.order)
			poly = legendre_polynomial(self.order-1)
			self.weights[1:self.order-1] *= 1/(poly.evaluate(self.nodes[1:self.order-1])**2)

		elif kind == 'legendre':
			self.nodes = legendre_polynomial(self.order).legendre_points()
			poly = legendre_polynomial(self.order).derivative()
			self.weights = 2/((1-self.nodes**2)*(poly.evaluate(self.nodes)**2))

	def integrate(self, func, round_off=20):
		return np.around(np.sum(self.weights*func(self.nodes)), round_off)
