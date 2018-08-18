import numpy as np
from matplotlib import pyplot as plt

class perceptron:
    def __init__(self,u,a,tset):
        self.umbral = u
        self.learningRate = a
        self.W = [0, 0]  #Peso w1 y w2 inicializados es 0  
        self.trainingset = tset
        self.x=np.arange(-5,10,0.1)
        self.point_separation()
        self.learn()

    #Separa las coordenadas en el eje Y y en eje X, esto para facilitar la representacion de los puntos 
    def point_separation(self): 
        self.c1x=[]
        self.c2y=[]
        self.c1y=[]
        self.c2x=[]
        for point, output in self.trainingset:
            if output==1:
                self.c1x.append(point[0])
                self.c1y.append(point[1])
            else:
                self.c2x.append(point[0])
                self.c2y.append(point[1])

    # suma ponderada x1*w1+x2*w2
    def weightedSum(self,values):
        return sum(value * weight for value, weight in zip(values, self.W))

    def drawrect(self):
        if self.W[0] or self.W[1]:
            if self.W[1]!=0:
                y=(self.umbral-self.x*self.W[0])/(self.W[1])
                plt.plot(self.x,y)
            else:
                plt.axvline(x=self.umbral/self.W[0]) 

    def drawpoints(self,x,y,symbol):
        plt.scatter(x,y, s=60, marker=symbol, linewidths=2)
        #plt.plot(c1x,c1y,'ob')
        #plt.plot(c2x,c2y,'og')

    def learn(self):
        while True:
            numerror = 0
            for point, output in self.trainingset:
                print(self.W)
                result = self.weightedSum(point) > self.umbral
                error = output - result
                if error != 0:
                    numerror += 1
                    for index, value in enumerate(point):
                        self.W[index] += self.learningRate * error * value
                    self.drawrect()
                    self.drawpoints(self.c1x,self.c1y,'+')
                    self.drawpoints(self.c2x,self.c2y,'_')
                    plt.show()
            if numerror == 0:
                break

    def evaluate(self,inputs):
        currentoutput=[]
        for x1,x2 in inputs:
            if x1*self.W[0]+x2*self.W[1]>self.umbral:
                currentoutput.append(1)
                self.drawpoints(x1,x2,'+')
            else:
                currentoutput.append(0)
                self.drawpoints(x1,x2,'_')
                self.drawrect()
        plt.show()
        return currentoutput


#Datos de entrenamiento
training=[([0,1],1), ([3,2],1), ([2,3],1), ([3,3],1),([0,-1],0),([-3,-2],0),([-2,-3],0)]
OR=[([0,0],0),([0,1],1),([1,1],1),([1,0],1)]
AND=[([0,0],0),([0,1],0),([1,1],1),([1,0],0)]
#Datos de prueba
test=[(0,0),(1,0),(1,1),(0,1),(2,2),(3,3),(-1,-1),(-2,3)]
test2=[(0,0),(1,1),(1,0),(0,1)]

#p1=perceptron(0.5,0.1,training)
#print (p1.evaluate(test))

p2=perceptron(0.5,0.1,OR)
print (p2.evaluate(test2))

#p3=perceptron(0.5,0.1,AND)
#print (p3.evaluate(test2))



