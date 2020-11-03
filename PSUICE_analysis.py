#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:14:48 2020

@author: trevorhillebrand
"""
import numpy as np
from netCDF4 import Dataset
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings

plt.rcParams.update({'font.size': 16}) # use size 16 font for all plots

def read_output(filename, fields='all', excludeFields=None):

    # Open output netCDF file in read-mode
    data = Dataset(filename, 'r') 
    data.set_auto_mask(False) # disable automatic masked arrays
    
    # Get variable names contained in file
    if fields == 'all':
        fieldsList = list(data.variables.keys())
        
    else:
        fieldsList = list(fields)
    
    
    # Remove fields specified to be excluded
    if excludeFields is not None:
        fieldsList.remove(excludeFields)
        
    # Create dictionaries to hold output data
    modelOutput = {}
    modelVarsInfo = {}
    # Loop through fields and load into pandas dataframe
    for field in fieldsList:
        modelOutput[field] = data.variables[field][:]
        
        # Get information about variable        
        modelVarsInfo[field] = {} #initialize dictionary
        modelVarsInfo[field]['longName'] = data.variables[field].long_name
        modelVarsInfo[field]['units'] = data.variables[field].units
        modelVarsInfo[field]['dimensions'] = data.variables[field].dimensions
        modelVarsInfo[field]['shape'] = data.variables[field].shape
    
    data.close()
    
    return modelOutput, modelVarsInfo  

def regrid_velocity(modelOutput, modelVarsInfo):
    
    # Calculate velocity fields. This is complicated by the use of y1,x0 for u-component
    # and y0,x1 for v-component. Use regrid_data to put these on the x1,y1 grid
    
    #regrid velocity variables onto x1, y1
    utopInterp = regrid_data('utop', modelOutput, modelVarsInfo)
    vtopInterp = regrid_data('vtop' ,modelOutput, modelVarsInfo)
    uaInterp = regrid_data('ua', modelOutput, modelVarsInfo)
    vaInterp = regrid_data('va', modelOutput, modelVarsInfo)
    ubotInterp = regrid_data('ubot', modelOutput, modelVarsInfo)
    vbotInterp = regrid_data('vbot', modelOutput, modelVarsInfo)
    
    modelOutput['surfaceSpeed'] = np.sqrt(utopInterp**2 + vtopInterp**2)
    
    modelOutput['depthAvgSpeed'] = np.sqrt(uaInterp**2 + vaInterp**2)
    
    modelOutput['basalSpeed'] = np.sqrt(ubotInterp**2 + vbotInterp**2)
    
    # Add information about velocity fields
    modelVarsInfo['surfaceSpeed'] = {}
    modelVarsInfo['surfaceSpeed']['longName'] = 'Ice velocity at surface, interpolated onto x1,y1 grid'
    modelVarsInfo['surfaceSpeed']['units'] = 'm/y'
    modelVarsInfo['surfaceSpeed']['dimensions'] = ('time', 'y1', 'x1')
    modelVarsInfo['surfaceSpeed']['shape'] = np.shape(modelOutput['surfaceSpeed'])
    
    modelVarsInfo['depthAvgSpeed'] = {}
    modelVarsInfo['depthAvgSpeed']['longName'] = 'Depth-averaged ice velocity, interpolated onto x1,y1 grid'
    modelVarsInfo['depthAvgSpeed']['units'] = 'm/y'
    modelVarsInfo['depthAvgSpeed']['dimensions'] = ('time', 'y1', 'x1')
    modelVarsInfo['depthAvgSpeed']['shape'] = np.shape(modelOutput['depthAvgSpeed'])
    
    modelVarsInfo['basalSpeed'] = {}
    modelVarsInfo['basalSpeed']['longName'] = 'Ice velocity at bottom, interpolated onto x1,y1 grid'
    modelVarsInfo['basalSpeed']['units'] = 'm/y'
    modelVarsInfo['basalSpeed']['dimensions'] = ('time', 'y1', 'x1')
    modelVarsInfo['basalSpeed']['shape'] = np.shape(modelOutput['basalSpeed'])
        
    return modelOutput, modelVarsInfo



def regrid_data(varName, modelOutput, modelVarsInfo, destX='x1', destY='y1'):
    #get x and y dimensions of variable to be interpolated
    sourceX = modelVarsInfo[varName]['dimensions'][2]
    sourceY = modelVarsInfo[varName]['dimensions'][1]
    varInterp = np.zeros(modelVarsInfo[varName]['shape'])
    iceMask = modelOutput['h']>1 # Create mask because interpolation will interpolate velocities onto ice-free areas
    # Loop through all timelevels and interpolate
    for time in range(0, len(modelOutput['time'])):
        varInterpolator = interpolate.interp2d(modelOutput[sourceX], modelOutput[sourceY], modelOutput[varName][time,:,:], kind='linear')
        varInterp[time,:,:] = varInterpolator(modelOutput[destX], modelOutput[destY])*iceMask[time,:,:]
    
    return varInterp


def get_variable_info(modelVarsInfo, varNames):

    if type(varNames) is str:
        varNames = [varNames] # convert string variable name to list for looping
    for varName in varNames:
        longName = modelVarsInfo[varName]['longName']
        units = modelVarsInfo[varName]['units']
        dims = modelVarsInfo[varName]['dimensions']
        shape = modelVarsInfo[varName]['shape']
        
        print("Info for variable {}:\nDescription: {}\nUnits: {}\nDimensions: {}\nShape:{}\n\n".format(
                varName, longName, units, dims, shape))

    
    
def plot_timeseries(modelOutput, modelVarsInfo, varNames):

    if type(varNames) is str:
        varNames = [varNames]
    fig, ax = plt.subplots(nrows=len(varNames))
    if len(varNames) == 1:
        ax.plot(modelOutput['time'], modelOutput[varName])
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(varName)
        
    if len(varNames) > 1:
        axCount=0
        for varName in varNames:
            ax[axCount].plot(modelOutput['time'], modelOutput[varName])
            ax[axCount].set_xlabel('Time (years)')
            ax[axCount].set_ylabel('{} ({})'.format(varName, modelVarsInfo[varName]['units']))
            axCount +=1
        fig.subplots_adjust(hspace=1)
        
    plt.show()
    
def plot_maps(modelOutput, modelVarsInfo, varName, timeLevel=-1, modelTime=None, logScale=False, cmap='Blues', maskIce=False, vmin=None, vmax=None):
   
    #get x and y dimensions of variable to be plotted
    x = modelOutput[modelVarsInfo[varName]['dimensions'][2]]
    y = modelOutput[modelVarsInfo[varName]['dimensions'][1]]
    xGrid, yGrid = np.meshgrid(x, y)  

    # Get appropriate time level
    if modelTime is not None:
        timeBool = (abs(modelTime-modelOutput['time']) == np.min(np.abs(modelTime-modelOutput['time']))) # boolean array
        timeLevel = [i for i, val in enumerate(timeBool) if val] # Get index of True in timeBool
        timeLevel = timeLevel[0]
 
    if logScale is False:
        var2plot = modelOutput[varName][timeLevel,:,:]
    else:
        with warnings.catch_warnings():  #ignore divide-by-zero warning when taking log10
            warnings.simplefilter("ignore")
            var2plot = np.log10(modelOutput[varName][timeLevel,:,:])
    
    # Set variable to nan where there is no ice, if desired. Just for plotting clarity.
    if maskIce is True:
        var2plot[modelOutput['h'][timeLevel,:,:] < 1] = np.nan 
        
        
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    
    varMap = ax.pcolormesh(xGrid, yGrid, var2plot, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('equal')
    ax.set_xlim(left=np.min(xGrid), right=np.max(xGrid))
    ax.set_xlabel('km', fontsize=18)
    ax.set_ylabel('km', fontsize=18)
    ax.set_ylim(bottom=np.min(yGrid), top=np.max(yGrid))
    
    ax.set_title("{} at time={} years".format(varName, modelOutput['time'][timeLevel]), fontsize=28)
    #sm = plt.cm.ScalarMappable(cmap=cmap, vim= , vmax= )
    #fig_abs.subplots_adjust(right=0.8)
    #cbar_ax = fig_abs.add_axes([])
    #cbarg = fig_abs.colorbar(sm, cax=cbar_ax, ticks = (some tuple))
    
    cbar = fig.colorbar(varMap)
    if logScale is False:
        cbar.set_label(label='{} ({})'.format(modelVarsInfo[varName]['longName'], modelVarsInfo[varName]['units']), fontsize=18)
    else: 
        cbar.set_label(label='{} (10$^x$ {})'.format(varName, modelVarsInfo[varName]['units']), fontsize=18)
    

    #plt.show()
    
    return fig, ax
    
    
def flowline(modelOutput, startX, startY, timeLevel=-1, max_iter = 1e5):
    x1 = modelOutput['x1']*1000.
    y1 = modelOutput['y1']*1000.
    x0 = modelOutput['x0']*1000.
    y0 = modelOutput['y0']*1000.
    
    
    # Interpolate velocities onto the same (x1, y1) grid
    vaInterpolator = interpolate.interp2d(x1, y0, 
                                          modelOutput['va'][timeLevel,:,:], kind='cubic')
    uaInterpolator = interpolate.interp2d(x0, y1, 
                                          modelOutput['ua'][timeLevel,:,:], kind='cubic')
    #maskwaterInterpolator = interpolate.interp2d(x1, y1,
                                                 #modelOutput['maskwater'][timeLevel,:,:], kind='linear')
    hInterpolator = interpolate.interp2d(x1,y1,
                                         modelOutput['h'][timeLevel,:,:], kind='cubic')
    
    flowlineX = np.zeros(int(max_iter),) # pre-allocate array for speed. This will be trimmed at the end, provided the loop doesn't reach max_iter
    flowlineY = np.zeros(int(max_iter),)
    
    flowlineX[0] = startX * 1.
    flowlineY[0] = startY * 1.
     

    flowlineIter=0
    
            
    dtFlowline = 10.
    
    #dist2GL = np.zeros(np.shape(modelOutput['time'])) # distance to grounding-line from startX, startY  along streamline for each time-level
    # loop through time-levels, calculate distance to grounding-line along a streamline
    # treat each time-level as steady flow.
    
    
    print("Performing flowline calculation for time-level {}".format(timeLevel))
    while np.min(x1) <= flowlineX[flowlineIter] <= np.max(x1) \
    and np.min(y1) <= flowlineY[flowlineIter] <= np.max(y1) \
    and flowlineIter <= max_iter-2 and hInterpolator(flowlineX[flowlineIter], flowlineY[flowlineIter]) >= 1.:
        flowlineIter += 1
        flowlineXOld = flowlineX[flowlineIter-1] * 1.
        flowlineYOld = flowlineY[flowlineIter-1] * 1.
        
        flowlineX[flowlineIter] = flowlineXOld + uaInterpolator(flowlineXOld, flowlineYOld) * dtFlowline
        flowlineY[flowlineIter] = flowlineYOld + vaInterpolator(flowlineXOld, flowlineYOld) * dtFlowline
        #dist2GL[timeInd] = dist2GL[timeInd] + np.sqrt((flowlineX[flowlineIter] - 
               #flowlineXOld)**2 + (flowlineY[flowlineIter] - flowlineYOld)**2)


    if flowlineIter >= max_iter:
        print('Reached maximum number of iterations but did not reach end of flowline')
        
    print("time-level {} took {} iterations.".format(timeLevel, flowlineIter))

    flowlineX = flowlineX[0:flowlineIter]
    flowlineY = flowlineY[0:flowlineIter]
    
    return(flowlineX, flowlineY)
   

def plot_groundingLine(modelOutput, ax, timeLevel=-1, color='black'):
    ax.contour(modelOutput["x1"], modelOutput["y1"], modelOutput["maskwater"][timeLevel,:,:], [0.5], colors=color)
    
def plot_velocityVectors(modelOutput, modelVarsInfo, ax, varNames=['ua', 'va'], timeLevel=-1, horzStep=5, color='black'):
    #First move both velocity vectors onto x1,y1
    uInterp = regrid_data(varNames[0], modelOutput, modelVarsInfo, destX='x1', destY='y1')
    vInterp = regrid_data(varNames[1], modelOutput, modelVarsInfo, destX='x1', destY='y1')
    
    xGrid, yGrid = np.meshgrid(modelOutput["x1"], modelOutput["y1"])
    
    ax.quiver(xGrid[1::horzStep, 1::horzStep], yGrid[1::horzStep, 1::horzStep], 
              uInterp[timeLevel,1::horzStep, 1::horzStep], vInterp[timeLevel,1::horzStep, 1::horzStep])               
  
def plot_transect(modelOutput, modelVarsInfo, plotVarName, ax=None, timeLevel=-1, transectX=None, transectY=None, method='linear',color='black'):
    x = modelOutput[modelVarsInfo[plotVarName]['dimensions'][2]]
    y = modelOutput[modelVarsInfo[plotVarName]['dimensions'][1]]
   
    plotVarInterpolator = interpolate.interp2d(x,y, modelOutput[plotVarName][timeLevel,: :], kind=method)
    distance = np.cumsum(np.sqrt(np.gradient(transectX)**2 + np.gradient(transectY)**2))
    plotVarInterp = transectX * np.nan
    for ii in np.arange(0,len(transectX)-1):
        plotVarInterp[ii] = plotVarInterpolator(transectX[ii], transectY[ii])
  
    if ax is not None: 
       ax.plot(distance, plotVarInterp, c=color)

    
    return(plotVarInterp, distance)
 
def timeseriesAtPoint(modelOutput, modelVarsInfo, varName, x, y, ax=None, interpMethod='linear'):
    #get x and y dimensions of variable to be interpolated
    sourceX = modelVarsInfo[varName]['dimensions'][2]
    sourceY = modelVarsInfo[varName]['dimensions'][1]
    varInterp = 0.0*modelOutput["time"]
    
    # Loop through all timelevels and interpolate
    for time in range(0, len(modelOutput['time'])):
        varInterpolator = interpolate.interp2d(modelOutput[sourceX], modelOutput[sourceY], modelOutput[varName][time,:,:], kind=interpMethod)
        varInterp[time] = varInterpolator(x, y)
        
    ax.plot(modelOutput["time"], varInterp)
    ax.set_ylabel(varName + ' (' + modelVarsInfo[varName]["units"] + ')')
    ax.set_xlabel("Time (yr)")
    
    return varInterp
    
    
def add_dHdt(modelOutput, modelVarsInfo):
    h = modelOutput["h"]
    time = modelOutput["time"]
    dHdt = np.zeros(np.shape(h))
    for timeLev in np.arange(1, np.shape(h)[0]):
        dHdt[timeLev,:,:] = (h[timeLev,:,:] - h[timeLev-1,:,:])/(time[timeLev] - time[timeLev-1])
    
    modelOutput["dHdt"] = dHdt
    modelVarsInfo['dHdt'] = {}
    modelVarsInfo['dHdt']['longName'] = 'Average rate of thickness change since last output time'
    modelVarsInfo['dHdt']['units'] = 'm/y'
    modelVarsInfo['dHdt']['dimensions'] = ('time', 'y1', 'x1')
    modelVarsInfo['dHdt']['shape'] = np.shape(dHdt)
    
    return modelOutput, modelVarsInfo

## TODO def movie()    
