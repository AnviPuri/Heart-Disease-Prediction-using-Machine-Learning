# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:30 2019

@author: Anvi Puri
"""

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Decision Tree', 'K-SVM', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Random Forest','ANN')
y_pos = np.arange(len(objects))
performance = [81.50,85.00,84.52,85.47,81.89,84.34,85.00]
 
plt.bar(y_pos, performance, align='center', alpha=0.5,color=['red','blue','green','yellow','orange','cyan','purple'])
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Algorithms')
 
plt.show()

