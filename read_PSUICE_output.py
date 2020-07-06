#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:04:03 2020
Reads in data from netCDF output file of PSU ice sheet model run. 
Default is to read in all field. Options to read in only certain fields,
or to exclude certain fields. 

@author: trevorhillebrand
"""

import numpy as np
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt


parser = OptionParser(description='Read PSU ice sheet model output')
parser.add_option("-f", "--filename", dest="filename", help="the netCDF output file")
parser.add_option("-v", "--variables", dest="fields", default="all", help="Model output fields to read")
parser.add_option("-x", "--exclude", dest="excludeFields", default=None, help="Model output fields to exclude")

for option in parser.option_list:
    if option.default != ("NO", "DEFAULT"):
        option.help += (" " if option.help else "") + "[default: %default]"
options, args = parser.parse_args()

# Open output netCDF file in read-mode
data = Dataset(options.filename, 'r') 
data.set_auto_mask(False) # disable automatic masked arrays

# Get variable names contained in file
if options.fields == 'all':
    fieldsList = list(data.variables.keys())
    
else:
    fieldsList = list(options.fields)


# Remove fields specified to be excluded
if options.excludeFields is not None:
    fieldsList.remove(options.excludeFields)
    
# Create dictionary to hold output data
modelOutput = {}

# Loop through fields and load into pandas dataframe
for field in fieldsList:
    modelOutput[field] = data.variables[field][:]
    
data.close()