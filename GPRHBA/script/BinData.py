import numpy as np
from astropy.cosmology import Planck15
from scipy.interpolate import interp1d

data = np.load('/home-4/wwong24@jhu.edu/work/dgerosa/sigmaspops/pretrack_reduce_time_uniform.npz',allow_pickle=True)

zAxis = np.power(10,np.linspace(np.log10(1e-8),np.log10(15),10000))
dL_data = Planck15.luminosity_distance(zAxis)
dL_interp = interp1d(zAxis,dL_data)

Mc_output = (data['Mtot_output']*(data['q_output']/(1+data['q_output'])**2)**(3./5))
#dL_output = []
#for i in data['z_output']:
#	dL_output.append(dL_interp(i))
q_output = data['q_output']
z_output = data['z_output']

mAxis = np.linspace(5,45,81)
#dLAxis = np.linspace(0,Planck15.luminosity_distance(1).value,41)
zAxis = np.linspace(0,1,81)
q_axis = np.linspace(0.1,1,81)
#chi_eff_axis = np.linspace(-1,1,41)

dataMat = []

M_bin_center = (mAxis[1:] + mAxis[:-1])/2
z_bin_center = (zAxis[1:] + zAxis[:-1])/2
q_bin_center = (q_axis[1:] + q_axis[:-1])/2
#chi_eff_bin_center = (chi_eff_axis[1:] + chi_eff_axis[:-1])/2

#grid = M_bin_center.reshape(40,1)
#grid = np.array(np.meshgrid(M_bin_center,dL_bin_center)).T.reshape(40*40,2)
grid = np.array(np.meshgrid(M_bin_center,z_bin_center)).T.reshape(80*80,2)
#grid = np.array(np.meshgrid(M_bin_center,q_bin_center)).T.reshape(40*40,2)
#grid = np.array(np.meshgrid(M_bin_center,chi_eff_bin_center)).T.reshape(40*40,2)
#grid = np.array(np.meshgrid(M_bin_center,z_bin_center,q_bin_center)).T.reshape(40*40*40,3)
#grid = np.array(np.meshgrid(M_bin_center,q_bin_center,chi_eff_bin_center)).T.reshape(40*40*40,3)


O1O2 = []
Design = []
Intrinsic = []

for i in [0,1,2,3,4,5,6]:
#	samples = [Mtot_output[i],z_output[i],q_output[i]]#,chi_eff_output[i]]
#	samples = [Mc_output[i],dL_output[i]]#,chi_eff_output[i]]

#	samples = [Mc_output[i]]
#	axes = [mAxis]
	samples = [Mc_output[i],z_output[i]]
	axes = [mAxis,zAxis]
#	samples = [Mc_output[i],q_output[i]]
#	axes = [mAxis,q_axis]

	O1O2.append(np.histogramdd(samples,bins=axes,weights=data['O1O2Rate'][i])[0].flatten())
	Design.append(np.histogramdd(samples,bins=axes,weights=data['designRate'][i])[0].flatten())
	Intrinsic.append(np.histogramdd(samples,bins=axes,weights=data['intrinsicRate'][i])[0].flatten())

np.savez('/home-4/wwong24@jhu.edu/scratch/BlackHoleKicks_HBA/simulation/pretrack_binned_time_uniform_Mc_z_design_smallBin',Observed=Design,Intrinsic=Intrinsic,hyperParameterDesign=data['sigma'][np.array([0,1,2,3,4,5,6])],histBins=grid)

