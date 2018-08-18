import numpy as np
from matplotlib import pyplot as plt

class perceptron:
    def __init__(self,u,a,tset):
        self.umbral = u             
        self.learningRate = a       #Tasa de aprendizaje
        self.W = [0, 0]  #Peso w1 y w2 inicializados es 0  
        self.trainingset = tset     #Conjunto de entrenamiento
        self.x=np.arange(-5,10,0.1)
        self.point_separation()
        self.learn()

    #Separa las componentes del eje x de las componentes del eje y, esto para facilitar la representacion de los puntos (x vs y)
    def point_separation(self): 
        self.c1x=[]
        self.c2y=[]
        self.c1y=[]
        self.c2x=[]
        for point, output in self.trainingset:
            if output==1:               #Puntos con salida 1
                self.c1x.append(point[0])           #Componentes en x
                self.c1y.append(point[1])           #Componentes en y
            else:                       #Puntos con salida 0
                self.c2x.append(point[0])           #Componentes en x
                self.c2y.append(point[1])           #Componentes en y

    # suma ponderada x1*w1+x2*w2
    def weightedSum(self,values):
        return sum(value * weight for value, weight in zip(values, self.W))

    #Dibuja la recta apartir de los pesos actuales
    def drawrect(self):
        if self.W[0] or self.W[1]:
            if self.W[1]!=0:
                y=(self.umbral-self.x*self.W[0])/(self.W[1])
                plt.plot(self.x,y)
            else:
                plt.axvline(x=self.umbral/self.W[0]) 

    #Dibuja los puntos cuya forma esta dada por symbol
    def drawpoints(self,x,y,symbol):
        plt.scatter(x,y, s=60, marker=symbol, linewidths=2)
        #plt.plot(c1x,c1y,'ob')
        #plt.plot(c2x,c2y,'og')

    def learn(self):
        while True:
            numerror = 0
            for point, output in self.trainingset:
                #print(self.W)
                result = self.weightedSum(point) > self.umbral          #Calcula la salida actual de la neurona
                error = output - result                                 #Calcula el error --> salida esperada - salida actual
                if error != 0:                                          #El perceptrón debe seguir entrenandose
                    numerror += 1
                    for index, value in enumerate(point):
                        self.W[index] += self.learningRate * error * value          #Actualización de los pesos
                    self.drawrect()
                    self.drawpoints(self.c1x,self.c1y,'+')
                    self.drawpoints(self.c2x,self.c2y,'_')
                    plt.show()
            if numerror == 0:                                           #Nuestro perceptron ha sido exitosamente entrenado
                break

    #Clasifica datos de prueba sin etiqueta
    def evaluate(self,inputs):
        currentoutput=[]
        for x1,x2 in inputs:
            if x1*self.W[0]+x2*self.W[1]>self.umbral:       
                currentoutput.append(1)                 #El punto pertenece a la clase 1
                self.drawpoints(x1,x2,'+')
            else:
                currentoutput.append(0)                 #El punto pertence a la clase 0
                self.drawpoints(x1,x2,'_')
                self.drawrect()
        plt.show()
        return currentoutput


#Datos de entrenamiento
training=[([0,1],1), ([3,2],1), ([2,3],1), ([3,3],1),([0,-1],0),([-3,-2],0),([-2,-3],0)]
OR=[([0,0],0),([0,1],1),([1,1],1),([1,0],1)]     #Compuerta or
AND=[([0,0],0),([0,1],0),([1,1],1),([1,0],0)]    #Compuerta and
#Datos de prueba
test=[(0,0),(1,0),(1,1),(0,1),(2,2),(3,3),(-1,-1),(-2,3)]
test2=[(0,0),(1,1),(1,0),(0,1)]

p1=perceptron(0.5,0.1,training)
print (p1.evaluate(test))

p2=perceptron(0.5,0.1,OR)
print (p2.evaluate(test2))

p3=perceptron(0.5,0.1,AND)
print (p3.evaluate(test2))



