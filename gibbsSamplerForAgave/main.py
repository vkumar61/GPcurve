#imports
import sampler
import readData

#hyper parameters
covL = 10
covLambda = 1

#number of samples to generate
nIter = 100

#load vectors from read csv File
dataVect, dataVectIndex, deltaT = readData.dataReader("C:/Users/vkuma/Downloads/data_to_share.txt")

#generate samples
sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)