# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:48:44 2018

@author: Pranav Rupireddy1
"""

from brian2 import *
import matplotlib.pyplot as plt
import numpy
#import sys

#Lines 217-218 is where you make the switch 


#Functions Used throughout program

##Initializing interactive plot with certain dimensions
#def InitPlot():
#    plt.ion()
#    plt.figure()
#    plt.axis((-5,105,-5,105))
#    plt.pause(2)


##Finds the time step for a sorted list of spike times
#def Finder(Number_You_R_Looking_For,Start_Index,Array):
#    i=0
#    indexMatrix=[]
#    TransformedNumber = int(10*(Number_You_R_Looking_For/ms))
#    for n in Array[Start_Index:]:
#        i=i+1
#        if (n) > (TransformedNumber):
#            break
#        elif (n) == (TransformedNumber): 
#            indexMatrix.append((i+Start_Index)-1)
#        else:
#            continue
#    return indexMatrix

#Reverse of the function above
#def ReverseFinder(Number_You_R_Looking_For,Array):
#    i=0
#    indexMatrix=[]
#    TransformedNumber = ((Number_You_R_Looking_For/ms))
#    print(TransformedNumber)
#    #print(Number_You_R_Looking_For)
#    if TransformedNumber == 707.7:
#        print("this should be the number")
#    else: 
#        1+1
#        #print("Not Recognized")
#        #print("This is the number "  + str(TransformedNumber))
#    for n in Array:
#        i=i-1
#        if Array[i] < (TransformedNumber):
#            if TransformedNumber == 707.7:
#                print(i)
#            break
#        elif Array[i] == (TransformedNumber): 
#            if TransformedNumber == 707.7:
#                print(Array[i])
#            indexMatrix.append(i)
#        else:
#            continue
#    return indexMatrix

#Reverse of the function above
def ReverseFinder(Number_You_R_Looking_For,Array):
    i=0
    indexMatrix=[]
    #print(Number_You_R_Looking_For)
    for n in Array:
        i=i-1
        if Array[i] < Number_You_R_Looking_For:
            1+1
            break
        elif Array[i] == Number_You_R_Looking_For: 
            indexMatrix.append(i)
        else:
            continue
    return indexMatrix

#Turns The indices for where a spike time occurs to the the actual neurons that spiked
def IndexToNeuron(IndicesInSpikeTime,SpikeIndexMatrix):
    TimeTNeuronSpikes = []
    for n in IndicesInSpikeTime:
        TimeTNeuronSpikes.append(SpikeIndexMatrix[n])
#    print(TimeTNeuronSpikes)
    return TimeTNeuronSpikes
#
##Takes Neural indices and converts to Locations
#def NeuronToLocation(TimeTNeuronSpikes,LocationMatrix):
#    TimeTNeuronLocation = []
#    for n in TimeTNeuronSpikes:
#        TimeTNeuronLocation.append(LocationMatrix[n])
#    return  TimeTNeuronLocation 

#Takes a Time Step and plots the neurons spiking if there are some

#def TimeStepToPlot(TimeStep, Start_Index, TransformedSpikeTime,SpikingIndices, LocationMatrix):
#    global Matrixi
#    Matrix1 = Finder(TimeStep, Start_Index, TransformedSpikeTime)
#    if len(Matrix1) == 0:
#        plt.suptitle("t = " + str(TimeStep))
#        plt.plot([0,0,0],[1,2,5],'w')
#        plt.axis((-5, 105, -5,105 ))
#        plt.pause(.01)
#        plt.cla()
#     
#        
#    else: 
#        Matrixi = Matrix1
#        Matrix2 = IndexToNeuron(Matrix1, SpikingIndices)
#        Matrix3 = NeuronToLocation(Matrix2, LocationMatrix)
#        Matrix4 = array(zip(*Matrix3))
#        plt.suptitle("t = "  + str(TimeStep))
#        plt.plot(Matrix4[0][:],Matrix4[1][:],'ro')
#        plt.axis((-5, 105,-5, 105))
#        plt.pause(.01)
#        plt.cla()
        
#Cuts off units from spike times and multiplies by 10, followed by int type case, for searching 
#def TransformSpikeMonT(SpikeTimeMatrix):
#    y= array((SpikeTimeMatrix)/(ms))
#    #x= y.astype(int)
#    return y

#First Iteration that searches for the first spike time
#
#def FirstIteration(SpikeTimeMatrix, SpikingIndices, LocationMatrix):
#    InitPlot()
#    global TransSpikeM 
#    TransSpikeM = TransformSpikeMonT(SpikeTimeMatrix)
#    FirstTime = SpikeTimeMatrix[0]
#    TimeStepToPlot(FirstTime, 0, TransSpikeM, spikemon.i, LocationMatrix)
# 
#Connectivity Visualizer
#def visualise_connectivity(S):
#    Ns = len(S.source)
#    Nt = len(S.target)
#    plt.figure(figsize=(10, 4))
#    plt.subplot(121)
#    plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
#    plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
#    for i, j in zip(S.i, S.j):
#       plt.plot([0, 1], [i, j], '-k')
#    plt.xticks([0, 1], ['Source', 'Target'])
#    plt.ylabel('Neuron index')
#    plt.xlim(-0.1, 1.1)
#    plt.ylim(-1, max(Ns, Nt))
#    plt.subplot(122)
#    plt.plot(S.i, S.j, 'ok')
#    plt.xlim(-1, Ns)
#    plt.ylim(-1, Nt)
#    plt.xlabel('Source neuron index')
#    plt.ylabel('Target neuron index')
    

#Synaptic Wiring Function from Adjacency Matrix
def SynapticConnector (Adjacency_Matrix,Synapses):
    rows,cols = nonzero(Adjacency_Matrix)
    Synapses.connect(i=rows,j=cols)

#Returns all of the spike times for the index of the neuron you are targetting
def SpikeTimeFinder(Index,SpikeIndices,SpikeTimes):
    mask = SpikeIndices == Index
    indices = where(mask)
    #print(indices[0])
    SpikeTimesForIndex = SpikeTimes[indices[0]]
    return SpikeTimesForIndex
    

#Importing Matrices


'''#Importing Adjacncency Matrix
print("Importing Adjacency")
Afile = sys.argv[1]
AMatrix = array(genfromtxt(Afile))'''

#AMatrix = array([[0,1,0,1,0],
#                 [1,0,1,0,1],
#                 [1,1,0,0,0], 
#                 [0,0,1,0,0],
#                 [0,1,0,1,0]])
#    
#N= AMatrix.shape[0]

N=1

#Importing Location Matrix
'''print("Importing Location")
Lfile = sys.argv[2]
L= array(genfromtxt(Lfile))'''

#L = array([[0,0],
#           [0,1],
#           [0,2],
#           [0,3],
#           [0,4]])


#PI Controller-Specific Functions

#ISI Matrix (changes it to only change for the first neuron:
global TrueISI
TrueISI = []

#Create PI Matrix (First Col = Accumulation, Second = Most Recent Spike Time, Third = Current Error)
global PIMatrix
PIMatrix = zeros((N,6))*second
global IMatrix
IMatrix = ones((N,1))*10*volt/second
IMatrix = IMatrix.flatten()

#PI Constants
tau = 100
alpha = 1 + 1/tau
Gain = -.4465*(second**2)/volt
P = (1/Gain)*(201*alpha - 1 + 20*((101*alpha**2 - alpha)**0.5))
print("This is P Before Implementaton " + str(P))
Int = 100*P
#Int = 100*P
DesiredISI = 0.01639344262295 *second


 #Returens the neurons that just spikeed   
def NeuronsThatSpiked(Number_You_R_Looking_For,Array, SpikingIndices):
    Matrix10 = ReverseFinder(Number_You_R_Looking_For,Array)
    Matrix11 = IndexToNeuron(Matrix10, SpikingIndices)     
    return Matrix11

#Modifies the PIMatrix's values for the neurons that have just spiked
def PIModifyEachSpike (NeuronsToUpdate,CurrentTime,SpikeTimes):
    for n in NeuronsToUpdate:
       # print("This is the current time" + str(CurrentTime))
       # print("This is the last spike time" + str(PIMatrix[n,1]))
        CurrentISI = CurrentTime - PIMatrix[n,1]
       # print("This is the ISI" + str(CurrentISI))
        #print("This is the previous sum of ISI's" + str(PIMatrix[n,4]))
        PIMatrix[n,4] = PIMatrix[n,4] + CurrentISI
        #print(PIMatrix[n,4])
       # print("This is the Sum of ISI Values" + str(PIMatrix[n,4]))
        PIMatrix[n,5] = PIMatrix[n,5] + 1*second
        #print("This is the Number of Spikes"  + str(PIMatrix[n,5]))
        PIMatrix[n,1] = CurrentTime
        
#Identifies the neurons that I need to change up the current into         
def NeuronsToModify(NeuronsToUpdate):
    NeuronsToMod = []
    for n in NeuronsToUpdate:
        if PIMatrix[n,1] >= (1000*ms + PIMatrix[n,3]):
            NeuronsToMod.append(n)
        else:
            continue
    return NeuronsToMod


def PIModifyEachPIUpdate (CurrentTime, NeuronsToModify, TargetISI):
    for n in NeuronsToModify:   
         NumberOfSpikes = PIMatrix[n,5]/second
         AverageISI = (PIMatrix[n,4])/NumberOfSpikes
         if n == 0:
             TrueISI.append(AverageISI)
         else:
             1+1
         CurrentError = (TargetISI - AverageISI)
         PIMatrix[n,0] = CurrentError
         TotalError = PIMatrix[n,2]  + CurrentError
         PIMatrix[n,2] = TotalError
         PIMatrix[n,3] = CurrentTime
         PIMatrix[n,4] = 0*second
         PIMatrix[n,5] = 0*second
     
#PI Control Matrices (Neurons to Update and the PI Matrix) Set up Function
def NeuronsGoodSpike(Number_You_R_Looking_For,Array, SpikingIndices,TargetISI):
    NeuronsToChange=[]
    Matrix12 = NeuronsThatSpiked(Number_You_R_Looking_For,Array, SpikingIndices)
    #print(Matrix12)
    for n in Matrix12:
        if PIMatrix[n,1] >= PIMatrix[n,3] + 200*ms:
            NeuronsToChange.append(n)
        else:
            PIMatrix[n,1] = Number_You_R_Looking_For
            continue
    #print(NeuronsToChange)
    return NeuronsToChange

#def horizontalstack(leftarray,rightarray):
#    rightarray = matrix(rightarray).T
#    leftarray = matrix(leftarray).T
#    NewMatrix = hstack((leftarray,rightarray))
#    return NewMatrix
  
   
#Neuron Setup    
             
   
#State Variables
eqs='''
dv/dt = (0.04/(1e-3*second)/((1e-3*volt)))*v**2 + (5/(1e-3*second))*v + 140*volt/second - u  + I :volt(unless refractory)
du/dt = a*(b*v - u)        :volt/second  (unless refractory)
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
G = NeuronGroup(N, eqs, threshold = 'v >= 30*mV', refractory = 2*1e-3*second, reset = reseqs, method = 'euler')

#Izhikevich Parameters
G.a = .01*1/(1e-3*second)
G.b = .585*1/(1e-3*second)  
G.c = -50*1e-3*volt
G.d = 3*volt/second
G.I = IMatrix

#Initial Voltage & Membrane Recovery Potential
G.v = -50*1e-3*volt
G.u = 0*volt/second

#Locaton Values
#G.x = (L[:,0]) * 1e-3*meter
#G.y = (L[:,1]) * 1e-3*meter 
#Settng up State Monitors
spikemon=SpikeMonitor(G)
M = StateMonitor(G,'v', record = True)
IMon = StateMonitor(G, 'I', record = True)
#Synaptic Setup


#Wiring Up Synapses
print("Wiring Up Synapses")
#S = Synapses(G,G, 'weight : volt', on_pre = 'v_post += weight')
#SynapticConnector(AMatrix,S)

#Synaptic Parameters
#S.delay = 1*1e-3*second
#S.weight= 100*1e-3*volt

#Actual Simulation

#visualise_connectivity(S)
#PI Controller
@network_operation(dt=.1*ms)
def change_I():
    #print(G.I)
    if not M.t:
        1+1
    elif M.t[-1] < 200*ms:
        #print("Still initial conditions")
        1+1
    else:
        print(M.t[-1])
        if not spikemon.t:
            #print("empty spike time matrix")
            1+1
        else: 
            SpikeModifyNeurons = NeuronsGoodSpike(M.t[-1],spikemon.t,spikemon.i,DesiredISI)
            PIModifyEachSpike(SpikeModifyNeurons, M.t[-1], spikemon.t)
            if SpikeModifyNeurons:
                print(SpikeModifyNeurons[0])
                print("This is last spike time" + str(PIMatrix[SpikeModifyNeurons[0],1]))
                print("This is the last modify time" + str(PIMatrix[SpikeModifyNeurons[0],3]))
            else:
                1+1
            NeuronsToMod = NeuronsToModify(SpikeModifyNeurons)
            #print(NeuronsToMod)
            PIModifyEachPIUpdate(M.t[-1], NeuronsToMod, DesiredISI)
            for n in NeuronsToMod:
                G.I[n] += .005*(Int*PIMatrix[n,2] + P*PIMatrix[n,0])
                print("Modification" + str(P*PIMatrix[n,2] + Int*PIMatrix[n,0]))

run(50000*ms)

#Plotting
'''
FirstIteration(spikemon.t, spikemon.i, L)
x = TransSpikeM[0]+1
for t in M.t[x:]:
    TimeStepToPlot(t, Matrixi[-1]+1, TransSpikeM, spikemon.i, L)

plt.clf()'''

#print (PIMatrix)

#print(G.I)

#y = len(TrueISI[0]) - 4
#print(range(1,y))
#
TrueISI = array(TrueISI)

SpikeTimesFor0Neuron = SpikeTimeFinder(0, spikemon.i, spikemon.t)


#Comparison = horizontalstack(SpikeTimesFor0Neuron, TrueISI[0])

#SpikeInter = matrix(SpikeTimesFor0Neuron).T
#TrueISIInter = matrix(TrueISI[0]).T

x = len(SpikeTimesFor0Neuron) 

mask = [where(M.t == SpikeTimesFor0Neuron[i])[0] for i in range(x)]
savetxt('Voltage at Spike Times.txt' , M.v[0, mask])

savetxt('TrueSpikesNeuron0.txt', SpikeTimesFor0Neuron)
savetxt('TrueISINeuron0.txt', TrueISI)
savetxt('G.I.txt',IMon.I[0])
savetxt('M.v values.txt', M.v[0])
#savetxt('SpikeTimesFor0Neuron,TrueISIfor0Neuron.txt', Comparison)

fig,axes = plt.subplots(2)

x = len(TrueISI)

print("This is spikemon.t" + str(spikemon.t))
print("This is the last spike time" + str(spikemon.t[-1]))
print("This is TrueISI" + str(TrueISI))

axes[0].plot(range(1,x+1),TrueISI[0:],label = 'ISI over time', color = 'blue', ls = '-')
axes[0].plot(range(1,x+1),ones(x)*0.01639344262295*second,label = 'ISI Target Value', color = 'red',ls='--')
axes[0].set_xlabel('Spike Bin Number')
axes[0].set_ylabel('ISI (seconds)')
axes[0].set_title('ISI over time')


axes[1].plot(IMon.t,IMon.I[0])
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Current Injected')
axes[1].set_title('Current Injected over time')

#axes.plot(M.t, M.v[0])
#axes.set_xlabel('Time (seconds)')
#axes.set_ylabel('Voltage')
#axes.set_title('Voltage of 0 Neuron over time')

#print(M.v[0, indices])


print(PIMatrix)
print(TrueISI[-1])

##Finshy things (Besides PI Control): 1) the voltage, 2) Synapses dont work 3) the 9.99 value
