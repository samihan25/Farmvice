# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

#ML libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
#

# Create your views here.
from django.http import HttpResponse

def index(request):
    return render(request, 'index.html')

def predict(request):
    nitrogen=float(request.POST["nitrogen"])
    phosphorus=float(request.POST["phosphorus"])
    potassium=float(request.POST["potassium"])
    pH_value=float(request.POST["pH_value"])
    season=request.POST["season"]
    district=request.POST["district"]

    #insert machine learning code here
    
    #n,p,k soil quality prediction code
    #start here
    df=pd.read_csv("npk.csv")

    X = df.iloc[:,:-1]
    Y = df.iloc[:,3]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    soil_quality=clf.predict([[nitrogen,phosphorus,potassium]])
    soil_quality=soil_quality[0]
    #ends here

    #crop prediction code
    #starts here
    import Orange

    data = Orange.data.Table("crop.tab")

    cn2_learner = Orange.classification.rules.CN2Learner()
    cn2_classifier = cn2_learner(data)

    c_values = data.domain.class_var.values

    input_district = district
    input_season = season
    input_soil_quality = soil_quality
    input_ph_4_45=0
    input_ph_45_5=0
    input_ph_5_55=0
    input_ph_55_6=1
    input_ph_6_65=1
    input_ph_65_7=0
    input_ph_7_75=0

    input_data=[[input_district,input_season,input_soil_quality,input_ph_4_45,input_ph_45_5,input_ph_5_55,input_ph_55_6,input_ph_6_65,input_ph_65_7,input_ph_7_75]]

    crop = c_values[int(cn2_classifier(input_data))]
    #ends here
    
    pesticide="abcd"

    return render(request,'predict.html',{'nitrogen':nitrogen, 'phosphorus':phosphorus, 'potassium':potassium, 'pH_value':pH_value, 'season':season, 'district':district, 'soil_quality':soil_quality, 'crop':crop, 'pesticide':pesticide})