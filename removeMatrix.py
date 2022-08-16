def removeMatrix(matrix):
	graphs = []
	observedIndecies = []
	for row in matrix:
		for column in row:
			if matrix[row][column] == 1:

	def buildGraph(index):
		hasSolution = False
		graph = []

	def checkValidity(index):
		if index in graphs:
			return graphs[index]
		elif index[0] + 1 > len(matrix)-1 or index[0]-1< 0 or index[1]+1 > len(matrix[0])-1 or index[1]-1 < 0:
			return True
		else:
			recursions = []
			if index[0]+1 <= len(matrix)-1:
				recursions.append((index[0]+1,index[1]))
			if index[0]-1 >= 0:
				recursions.append((index[0]-1,index[1]))
			if index[1]+1 <= len(matrix[0])-1:
				recursions.append(index[0],index[1]+1)
			if index[1]-1 >= 0:
				recursions.append(index[0],index[1]-1)
			for recurse in recursions:
				return 



# class graph():
# 	def __init__(self):
# 	self.index
# 	self.left
# 	self.right				

matrix = [
[1,0,0,0,0,0],
[0,1,0,1,1,1],
[0,0,1,0,1,0],
[1,1,0,0,1,0],
[1,0,1,1,0,0],
[1,0,0,0,0,1]
]