# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:20:20 2018

@author: Pranav Rupireddy1
"""

'''Instructions: Go to Command Prompt, Type in "cd Desktop", Type in anapath.bat, 
Type in cd Path (You put the path of the directory here), Type in python BasicSimulation.py
NameofAdjacencyMatrix.txt NameofLocatonMatrix.txt'''

#importing Libraries
from brian2 import *
import matplotlib.pyplot as plt
import numpy
import sys

WeightMatrix= linspace(0,20, num = 100)

#Initializing interactive plot with certain dimensions
def InitPlot():
    plt.ion()
    plt.figure()
    plt.axis((-5,105,-5,105))
    plt.pause(2)


#Finds the time step for a sorted list of spike times
def Finder(Number_You_R_Looking_For,Start_Index,Array):
    i=0
    indexMatrix=[]
    TransformedNumber = int(10*(Number_You_R_Looking_For/ms))
    for n in Array[Start_Index:]:
        i=i+1
        print(n)
        print("This is Transformed Number")
        print(TransformedNumber)
        if (n) > (TransformedNumber):
            print("break")
            break
        elif (n) == (TransformedNumber): 
            print("Appended")
            indexMatrix.append((i+Start_Index)-1)
        else:
            print("continue")
    return indexMatrix


#Turns The indices for where a spike time occurs to the the actual neurons that spiked
def IndexToNeuron(IndicesInSpikeTime,SpikeIndexMatrix):
    TimeTNeuronSpikes = []
    for n in IndicesInSpikeTime:
        TimeTNeuronSpikes.append(SpikeIndexMatrix[n])
    return TimeTNeuronSpikes

#Takes Neural indices and converts to Locations
def NeuronToLocation(TimeTNeuronSpikes,LocationMatrix):
    TimeTNeuronLocation = []
    for n in TimeTNeuronSpikes:
        TimeTNeuronLocation.append(LocationMatrix[n])
    return  TimeTNeuronLocation 

#Takes a Time Step and plots the neurons spiking if there are some
def TimeStepToPlot(TimeStep, Start_Index, TransformedSpikeTime,SpikingIndices, LocationMatrix):
    global Matrixi
    Matrix1 = Finder(TimeStep, Start_Index, TransformedSpikeTime)
    if len(Matrix1) == 0:
        print("End")   
        print("Matrixi is before iteration ")
        print(Matrixi) 
        plt.suptitle("t = " + str(TimeStep))
        plt.plot([0,0,0],[1,2,5],'w')
        plt.axis((-5, 105, -5,105 ))
        plt.pause(.01)
        plt.cla()
     
        
    else: 
        print("plotting")
        Matrixi = Matrix1
        print("Matrixi is during new iteration ")
        print(Matrixi)
        Matrix2 = IndexToNeuron(Matrix1, SpikingIndices)
        print(Matrix2)
        Matrix3 = NeuronToLocation(Matrix2, LocationMatrix)
        print(Matrix3)
        Matrix4 = array(zip(*Matrix3))
        print(Matrix4)
        plt.suptitle("t = "  + str(TimeStep))
        plt.plot(Matrix4[0][:],Matrix4[1][:],'ro')
        plt.axis((-5, 105,-5, 105))
        plt.pause(.01)
        plt.cla()
        
#Cuts off units from spike times and multiplies by 10, followed by int type case, for searching 
def TransformSpikeMonT(SpikeTimeMatrix):
    y= array(10*((SpikeTimeMatrix)/(ms)))
    x= y.astype(int)
    return x

#First Iteration that searches for the first spike time
def FirstIteration(SpikeTimeMatrix, SpikingIndices, LocationMatrix):
    InitPlot()
    global TransSpikeM 
    TransSpikeM = TransformSpikeMonT(SpikeTimeMatrix)
    FirstTime = SpikeTimeMatrix[0]
    TimeStepToPlot(FirstTime, 0, TransSpikeM, spikemon.i, LocationMatrix)

#Synaptic Wiring Function from Adjacency Matrix
def SynapticConnector (Adjacency_Matrix,Synapses):
    rows,cols = nonzero(Adjacency_Matrix)
    Synapses.connect(i=rows,j=cols)

print("About to import Adjacency")
#Importing Adjacncency Matrix
Afile = sys.argv[1]
AMatrix = array(genfromtxt(Afile))
N= AMatrix.shape[0]

print("About to import Location")
#Importing Location Matrix
Lfile = sys.argv[2]
L= array(genfromtxt(Lfile))

                
#State Variables
eqs='''
dv/dt = (0.04/(1e-3*second)/((1e-3*volt)))*v**2 + (5/(1e-3*second))*v + 140*volt/second - u  + I  :volt(unless refractory)
du/dt = a*(b*v - u)        :volt/second (unless refractory)
a                                 :1/second
b                                 :1/second 
c                                 :volt 
d                                 :volt/second
I                                 :volt/second
x                                 :meter
y                                 :meter
'''

#Reset Equations
reseqs='''
v = c   
u = u+d 
'''

                
#Creating the Neuron Group
G = NeuronGroup(N, eqs, threshold = 'v >= 30*mV', refractory = 0*1e-3*second, reset = reseqs, method = 'euler')

#Izhikevich Parameters
G.a = .02*1/(1e-3*second)
G.b = .2*1/(1e-3*second) 
G.c = -65*1e-3*volt
G.d = 8*volt/second

#Initial Voltage & Membrane Recovery Potential
G.v = -65*1e-3*volt
G.u = 0*volt/second

#Locaton Values
G.x = (L[:,0]) * 1e-3*meter
G.y = (L[:,1]) * 1e-3*meter 

G.run_regularly('I = rand()*10.5*volt/second', dt=1*ms)

#Wiring Up Synapses
print("About to Wire")
S = Synapses(G,G, 'weight : volt', on_pre = 'v_post += weight')
SynapticConnector(AMatrix,S)
print("Finished Wiring")

#Synaptic Parameters
S.delay = 1*1e-3*second

#Settng up State Monitors
spikemon=SpikeMonitor(G)
M = StateMonitor(G,'v', record = True)

store()
#Setting up and running the network
for x in WeightMatrix:
    restore()
    S.weight=x*1e-3*volt
    run(100*ms)
    #Actual Program
    FirstIteration(spikemon.t, spikemon.i, L)
    x = TransSpikeM[0]+1
    for t in M.t[x:]:
        TimeStepToPlot(t, Matrixi[-1]+1, TransSpikeM, spikemon.i, L)

    plt.clf()
