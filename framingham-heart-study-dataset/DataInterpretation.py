# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:18:38 2019

@author: Anvi Puri
"""

#Analysing the Dataset
def draw_histograms(dataset, features, row, col):
    fig=plt.figure(figsize=(300,500))
    for i, feature in enumerate(features): # It allows us to loop over something and have an automatic counter
        lay=fig.add_subplot(row,col,i+1)
        dataset[feature].hist(bins=20,ax=lay,facecolor='yellow')
        lay.set_title(feature+" Distribution",color='blue')
      
    fig.tight_layout()  
    plt.show()
draw_histograms(dataset,dataset.columns,4240,15)