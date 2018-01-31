

# ------------------------------
# - read the input data:

import numpy as np
import matplotlib.pyplot as plt
import network
import catalogs_loader
#-------------------------------

# Catalogs_loader fills in the flux and results of the training and tests arrays

(training_data, test_data) = catalogs_loader.load_catalogs()

training_data = list(training_data)

net = network.Network([40, 3, 2])

epoch= 0
LR = 1.5
x = []
y = []

net.SGD(training_data, 40, 15, LR, test_data=test_data) 

