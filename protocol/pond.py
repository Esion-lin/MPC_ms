def dispatch(cls, value:IntTensor):
		'''
		该函数返回一个tuple其中第一位为自己持有的share， 第二位list为其他人的share -> （自己的share，[player1的share，player2的share...]）
		'''
		value0 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		#value1 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		#TODO module addition 
		value1 = value - value0
		return (value0, [value1])