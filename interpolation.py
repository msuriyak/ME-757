from __future__ import division
import numpy as np

class lagrange_1d(object):
	x_vals = None
	func_vals = None
	def __init__(self, x_vals, func_vals=None):
		self.x_vals = x_vals
		self.order = len(x_vals)   				# N + 1
		self.basis_functions = [self.basis(j) for j in range(self.order)]
		self.deriv_basis = [self.derivative(j) for j in range(self.order)]
		
		if func_vals != None:
			self.func_vals = func_vals

	def fit_lagrange(self, func_vals):
		self.func_vals = func_vals

	def basis(self, j):
		den = np.prod([(self.x_vals[j] - self.x_vals[i]) for i in range(self.order) if i!=j], axis=0)
		def func_j(x):
			x = np.array(x)
			res = np.prod([(x - self.x_vals[i]) for i in range(self.order) if i!=j], axis=0)
			return res/den
		return func_j

	def evaluate(self, x, round_off=18):
		res = 0
		for i, func in enumerate(self.basis_functions):
			res += func(x)*self.func_vals[i]
		return np.around(res, round_off)

	def derivative(self, j):
		den =  np.prod([(self.x_vals[j] - self.x_vals[i]) for i in range(self.order) if i!=j])
		def deriv_j(x):
			res = 0.0
			for m in range(self.order):
				if m!=j:
					res += np.prod([(x - self.x_vals[i]) for i in range(self.order) if i!=j and i!=m], axis=0)
			return res/den
		return deriv_j	

	def evaluate_derivative(self, x, round_off=32):
		res = 0
		for i, func in enumerate(self.deriv_basis):
			res += func(x)*self.func_vals[i]
		return np.around(res, round_off)

	def make_Lij(self, sampling_points, round_off=12):
		return np.around(np.array([[self.basis_functions[i](elem) for elem in sampling_points] for i in range(self.order)]), round_off)

if __name__ == '__main__':
	print('Running main')
