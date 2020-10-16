class Triple_generator:
	#乘法triple
	@classmethod
	def triple(cls, shape:list):
		a = IntTensor(Factory.get_uniform(shape), internal = True)
		b = IntTensor(Factory.get_uniform(shape), internal = True)
		c = a * b
		return [a,b,c]
	#矩阵乘法triple
	@classmethod
	def mat_triple(cls, shapeX:list, shapeY:list):
		def check_mat(shapeA, shapeB):
			if len(shapeA) != len(shapeB) or shapeA[-1] != shapeA[-2]:
				return False
			return True
		if check_mat(shapeX, shapeY):
			a = IntTensor(Factory.get_uniform(shapeX), internal = True)
			b = IntTensor(Factory.get_uniform(shapeY), internal = True)
			c = a.Matmul(b)
			return [a,b,c]
		else:
			raise TypeError("shapes do not match!{} - {}".format(shapeX, shapeY))

	#卷积triple
	@classmethod
	def conv_triple(cls, shapeX:list, shapeY:list, stride, padding):
		a = IntTensor(Factory.get_uniform(shapeX), internal = True)
		b = IntTensor(Factory.get_uniform(shapeY), internal = True)
		c = a.Conv(b,stride,padding)
		return [a,b,c]
	#square triple
	@classmethod
	def square_triple(cls, shape:list,):
		a = IntTensor(Factory.get_uniform(shape), internal = True)
		b = a*a
		return [a, b]