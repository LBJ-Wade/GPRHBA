import matplotlib.pyplot as plt
import numpy as np
from GWcalculator.core import graphicsSettings
from InterpUnit import InterpUnit
import os
import argparse


# ===================================================
# Define parser and get parser variable
# ===================================================
parser = argparse.ArgumentParser(description='Create injection')
parser.add_argument('--SimFile',type=str,help='Simulation file',required=True)
#parser.add_argument('--OutputDir',type=str,help='Output directory',required=True)
parser.add_argument('--Tobs',type=float,help='Number of expected event',default=1,required=False)
parser.add_argument('--Tag',type=str,help='Tag of files',default='Injection',required=False)

args = parser.parse_args()
SimFile = args.SimFile
#OutputDir = args.OutputDir
Tobs = args.Tobs
Tag = args.Tag

# ===================================================
# Loading training data and construct Gaussian Processes
# ===================================================

PCAtolerance = 1e-20
simulations = np.load(SimFile)
#output_directory = OutputDir
tag = Tag
sigma = 100
#if not os.path.exists(output_directory):
#	os.mkdir(output_directory)

histBins = simulations['histBin']
if histBins.ndim ==1:
	histBins = histBins.reshape(histBins.size,1)

Intrinsic = simulations['Intrinsic']*Tobs
hyperParameterDesign = simulations['hyperParameterDesign']
Intrinsic = np.log10(np.array(Intrinsic).T)
Intrinsic[np.isinf(Intrinsic)] = -20
Interp_intrinsic = InterpUnit(Intrinsic,histBins,np.array([hyperParameterDesign]).T,tol=PCAtolerance)

# ===================================================
# Once the GPR unit is trained, we can use it to interpolate stuff.
# ===================================================

print('\nPrinting the interpolated result at sigma=100:')
print('\nRate:')
print(Interp_intrinsic.GetNormalization([100]))
print('\n2D distribution:')
print(Interp_intrinsic.GetDistribution([100]))
print('\nNumber of event in the bin center at (Mc=10,z=0.5):')
print(Interp_intrinsic.GetNtheta([10,0.5],[100]))

