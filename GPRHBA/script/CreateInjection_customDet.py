import matplotlib.pyplot as plt
import numpy as np
from GWcalculator.core import graphicsSettings
from InterpUnit import InterpUnit
import os
import argparse

parser = argparse.ArgumentParser(description='Create injection')
parser.add_argument('--SimFile',type=str,help='Simulation file',required=True)
parser.add_argument('--OutputDir',type=str,help='Output directory',required=True)
parser.add_argument('--Tobs',type=int,help='Number of expected event',required=True)
parser.add_argument('--Tag',type=str,help='Tag of files',default='Injection',required=False)

args = parser.parse_args()
SimFile = args.SimFile
OutputDir = args.OutputDir
Tobs = args.Tobs
Tag = args.Tag

# ===================================================
# Load training data and construct Gaussian Processes
# ===================================================

PCAtolerance = 1e-20
simulations = np.load(SimFile)
output_directory = OutputDir
tag = Tag
sigma = 100
if not os.path.exists(output_directory):
	os.mkdir(output_directory)

histBins = simulations['histBins']
if histBins.ndim ==1:
	histBins = histBins.reshape(histBins.size,1)

Intrinsic = simulations['Intrinsic']*Tobs
hyperParameterDesign = simulations['hyperParameterDesign']
Intrinsic = np.log10(np.array(Intrinsic).T)
Intrinsic[np.isinf(Intrinsic)] = -20
Interp_intrinsic = InterpUnit(Intrinsic,histBins,np.array([hyperParameterDesign]).T,tol=PCAtolerance)

from scipy.stats import rv_histogram
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from GWcalculator.core.samplingMethod import inverseTransformSampling

posterior_point = 300
#samplePoint = np.random.poisson(Nevent)
#print(samplePoint)
#Tobs = float(Nevent)/int(Interp_observed.GetNormalization([sigma]))
import pickle
from GWcalculator.core.commonFunctions import *


pkl_file = open('./redump.pkl','rb')
pdet_f = pickle.load(pkl_file)
BinSize = np.array([np.diff(np.unique(histBins.T[0]))[0],np.diff(np.unique(histBins.T[1]))[0]])/2
probability = Interp_intrinsic.GetDistribution([sigma])/Interp_intrinsic.GetNormalization([sigma])
posterior_vector = []

def GenSamples(size):
	histSamples = histBins[np.random.choice(np.arange(histBins.shape[0]),p=probability,size=size)]
	posterior_buffer = np.zeros((size,2))
	posterior_buffer[:,0] = np.random.uniform(histSamples.T[0]- BinSize[0],histSamples.T[0]+BinSize[0])
	posterior_buffer[:,1] = np.random.uniform(histSamples.T[1]- BinSize[1],histSamples.T[1]+BinSize[1])
	pdet_array = pdet_f(np.insert(np.array(McEtatoM1M2(posterior_buffer.T[0],0.25)),2,posterior_buffer.T[1],axis=0).T)
	accepted_samples = posterior_buffer[np.where(pdet_array>np.random.uniform(size=pdet_array.size))[0]]
	return accepted_samples

posterior_buffer = GenSamples(np.random.poisson(Interp_intrinsic.GetNormalization([sigma])))
print(posterior_buffer.shape[0])
posterior_buffer_long = np.random.normal(posterior_buffer,np.array([0.01*posterior_buffer.T[0],0.01*posterior_buffer.T[1]]).T,size=(posterior_point,posterior_buffer.shape[0],histBins.shape[1])).T
posterior_vector = np.array(posterior_buffer_long.reshape(histBins.shape[1],posterior_buffer.shape[0]*posterior_point)).T

size = [posterior_point for i in range(posterior_buffer.shape[0])]
length_array = np.insert(np.cumsum(size),0,0)
prior_vector = np.ones(posterior_vector.size)

#def LogLikelihood(sigma):
#	Ndet = np.sum(pdet_f(np.insert(McEtatoM1M2(histBins.T[0],0.25),2,histBins.T[1],axis=0).T)*Interp_intrinsic.GetDistribution([sigma]))
#	Int = np.sum(np.log(Interp_intrinsic.GetNtheta(posterior_buffer,[sigma])))
#	return Int-Ndet

def LogLikelihood(cube):
	if type(cube) is not np.ndarray:
		cube = np.array([cube])
	p_theta = Interp_intrinsic.GetNtheta(posterior_vector,cube)
	Ndet = np.sum(pdet_f(np.insert(McEtatoM1M2(histBins.T[0],0.25),2,histBins.T[1],axis=0).T)*Interp_intrinsic.GetDistribution(cube))
	single_mean = np.zeros(length_array.size-1)
	for i in range(length_array.size-1):
	    single_mean[i] = np.mean(p_theta[length_array[i]:length_array[i+1]]/prior_vector[length_array[i]:length_array[i+1]])
	return np.sum(np.log(single_mean[np.nonzero(single_mean)]))-Ndet

import emcee
from multiprocessing import Pool

def LogPrior(cube):
  if (cube<0)+(cube>265):
    return -np.inf
  else:
    return np.log10(1./265)

LogLikelihood = np.vectorize(LogLikelihood)
LogPrior = np.vectorize(LogPrior)
def LogProb(cube):
  return LogLikelihood(cube)+LogPrior(cube)



nwalkers,ndim = 10,1
p0 = np.array([np.random.uniform(0,265,nwalkers)]).T
pool = Pool(4)
sampler = emcee.EnsembleSampler(nwalkers,ndim,LogProb,pool=pool)
p0,_,_ = sampler.run_mcmc(p0,100)
sampler.reset()
p0,_,_ = sampler.run_mcmc(p0,100)

np.savez(OutputDir,samples=sampler.flatchain,Tobs=Tobs,Ndet=Interp_intrinsic.GetNormalization([sigma]),sigma=sigma)
