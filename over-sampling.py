#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:48:43 2018

@author: nilpuig
"""

y = label_TRN
X = FluxesTRN

# Upsampling . Please note that stars data was considered 4 times as there was an inherent bias within the data distribution
frames=[starTraining,galaxyTraining,starTraining,starTraining,starTraining]
combined=pd.concat(frames)

# Randomizing the order of the training data so as to ensure a smooth training process
combined=combined.sample(frac=1)
test=pd.concat([testStar,testGalaxy])
