#read file to get data

#def gatherData(filename) --> list()

#def createValidationSet --> list()
		#remove from training data and create a validation set

#def create y list() --> list()
		# seperate target y data from x data
import numpy as np
import math
import random
import matplotlib.pyplot as plt
def gatherData(filename):
    with open(filename+'.data') as f:
        rows = f.readlines()
    data = []
    dataY = []
    for row in rows:
        splitRow = row.split(",")
        dataX = []
        for i in range(len(splitRow)):
            if i != len(splitRow)-1:
                if splitRow[i] != "?":
                    dataX.append(float(splitRow[i]))
                else:
                    dataX.append(splitRow[i])
            else:
                YVal = float(splitRow[i][0])
                if YVal < 1:
                    dataY.append(-1)
                else:
                    dataY.append(1)
                data.append(dataX)
    return (data,dataY)

def getMeans(data):
    means = []
    for i in range(len(data[0])):
        s = 0
        for d in data:
            if d[i] != "?":
                s+= float(d[i])
        means.append(round(s/len(data),3))
    return means


def fillData(data):
    means = getMeans(data)
    for d in data:
        for i in range(len(d)):
            if d[i] == "?":
                d[i] = means[i]


class LinRegPredictor(object):
    def __init__(self,eta,dataX,dataY,valX,valY,maxEpochs):
        #self.weights = [0 for i in range(len(dataX[0]))]
        self.weights = np.zeros(len(dataX[0]))
        #self.weights = np.array([random.uniform(-100,100) for i in range(len(dataX[0]))])

        self.features = []
        self.eta = eta
        self.dataX = dataX
        self.dataY = np.array(dataY)
        self.valX = valX
        self.valY = np.array(valY)
        self.maxEpochs = maxEpochs
        self.scoreOverEpochs = []



    def trainHingeLoss(self):
        epochs = 0
        while epochs < self.maxEpochs:
            avgscore = []
            for i in range(len(self.dataX)):
                margin = (np.dot(self.weights,np.array(self.dataX[i]))*self.dataY[i])
                if margin >= 1:
                    gradient = 0
                else: 
                    gradient = -1*np.array(self.dataX[i])*self.dataY[i]
                if epochs%1 == 0:
                    #print("SIGMOID: ",sigmoid)
                    score = max(0,1-margin)
                    avgscore.append(score)
                self.weights -= self.eta*gradient
                #print("WEIGHTS: ",self.weights)
            accuracy = self.predict()[1]
            if epochs % 1 == 0:
                scoreVal = accuracy
                self.scoreOverEpochs.append(scoreVal)
            if epochs % 1000 == 0 and self.eta >= 0.001:
                self.eta/=10
            epochs +=1
    def sigmoid(self,x):
    	return 1/(1+np.exp(x))
    def trainLogLoss(self):
        epochs = 0
        #use correlations as initial weights
        #self.weights = self.getCorrelations()
        oldGradient = None
        while epochs < self.maxEpochs:
            avgscore = []
            for i in range(len(self.dataX)):
                margin = np.dot(self.weights,np.array(self.dataX[i]))
                sigmoid = self.sigmoid(margin)
                if self.dataY[i] == 0:
                	y = 0
                else:
                	y = 1
                score = (y*np.log(sigmoid) + (1-y)*np.log(1-sigmoid))
                avgscore.append(score)
                gradient = np.array(self.dataX[i]).T * (sigmoid - self.dataY[i])
                #gradient = (margin -self.dataY[i])/(self.weights*(1-margin))#-1*np.array(self.dataX[i])*self.dataY[i]
                self.weights -= self.eta*gradient
            avgscore = np.sum(avgscore)/len(avgscore)
                #print("WEIGHTS: ",self.weights)

            self.scoreOverEpochs.append(avgscore)
            if epochs % 100 == 0 and self.eta >= 0.001:
                self.eta/=10
            if epochs > 1 and np.linalg.norm(self.weights - oldGradient) < 0.00001:
            	print("FINSHED IN: ", epochs, " EPOCHS")
            	break
            oldGradient = self.weights
            epochs +=1
    

    def predict(self):
        predictions = []
        #print("FINAL WEIGHTS: ",self.weights)
        for dataPoint in self.valX:
            score = np.dot(self.weights,dataPoint)
            pred = self.sigmoid(score)
            #print(score)
            if pred >0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        correct = 0
        for p in range(len(predictions)):
            if predictions[p] == valY[p]:
                correct+=1
        accuracy = correct/len(predictions)
        return (predictions,accuracy)
    def predictUsingCorrelations(self):
        predictions = []
        #print("FINAL WEIGHTS: ",self.weights)
        for dataPoint in self.valX:
            score = np.dot(self.getCorrelations()*5000,dataPoint)
            #print(score)
            if score >0:
                predictions.append(1)
            else:
                predictions.append(-1)
        return predictions  

    def getCorrelations(self):
        columns = []
        correlations = []
        for i in range(len(self.dataX[0])):
            column = []
            for d in self.dataX:
                column.append(d[i])
            columns.append(column)

        for column in columns:
            correlation = np.corrcoef(column,self.dataY)
            #print("CORRELATION: ",correlation[0][1])
            correlations.append(correlation[0][1])
        return np.array(correlations)


data = gatherData("diabetes")
fillData(data[0])
stopData = int(len(data[0])*0.8)
trainSet = [data[0][i] for i in range(stopData)]
valSet = [data[0][i] for i in range(stopData,len(data[0]))]
trainY = [data[1][i] for i in range(stopData)]
valY = [data[1][i] for i in range(stopData,len(data[0]))]

linRegLearner = LinRegPredictor(0.1,trainSet,trainY,valSet,valY,1000)
print("Correlations: ",linRegLearner.getCorrelations())
linRegLearner.trainLogLoss()
#linRegLearner.trainHingeLoss()
predictions = linRegLearner.predict()
# #predictions = linRegLearner.predictUsingCorrelations()


print("ACTUAL: ",valY)
print("PREDIC: ",predictions)
correct = 0
predictionArray = predictions[0]
for p in range(len(predictionArray)):
    if predictionArray[p] == valY[p]:
        correct+=1
print("TOTAL CORRECT: ",correct,"/",len(predictionArray))
print("WEIGHTS: ",linRegLearner.weights)

xAxis = [i for i in range(len(linRegLearner.scoreOverEpochs))]
f = max(linRegLearner.scoreOverEpochs)
index = linRegLearner.scoreOverEpochs.index(f)
print("OPTIMAL EPOCHS = ",index+1," at accuracy: ",linRegLearner.scoreOverEpochs[index])
plt.plot(xAxis, linRegLearner.scoreOverEpochs)
plt.show()

# i = 0
# for v in valSet:
#     print("Example: ",v, "-- Prediction: ",predictions[i])
#     i+=1






	

        