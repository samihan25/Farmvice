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
    nitrogen = float(request.POST["nitrogen"])
    phosphorus = float(request.POST["phosphorus"])
    potassium = float(request.POST["potassium"])
    
    district = request.POST["district"]
    season = request.POST["season"]
    pH_value = float(request.POST["pH_value"])

    #insert machine learning code here

    #n,p,k soil quality prediction code
    #start here
    soil_quality = soil_quality_fun(nitrogen,phosphorus,potassium)
    #ends here

    #crop prediction code
    #starts here
    crop = crop_predict(district,season,pH_value,soil_quality)
    #ends here
    
    pesticide="wxyz"

    return render(request,'predict.html',{'nitrogen':nitrogen, 'phosphorus':phosphorus, 'potassium':potassium, 'pH_value':pH_value, 'season':season, 'district':district, 'soil_quality':soil_quality, 'crop':crop, 'pesticide':pesticide})

def soil_quality_fun(nitrogen,phosphorus,potassium):
    df=pd.read_csv("npk.csv")

    X = df.iloc[:,:-1]
    Y = df.iloc[:,3]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    soil_quality=clf.predict([[nitrogen,phosphorus,potassium]])
    return soil_quality[0]

def crop_predict(district,season,pH_value,soil_quality):
    import Orange

    data = Orange.data.Table("crop.tab")

    cn2_learner = Orange.classification.rules.CN2Learner()
    cn2_classifier = cn2_learner(data)

    c_values = data.domain.class_var.values

    input_district = district
    input_season = season
    input_soil_quality = soil_quality
    input_ph_4_45, input_ph_45_5, input_ph_5_55, input_ph_55_6, input_ph_6_65, input_ph_65_7, input_ph_7_75 = ph_fun(pH_value)

    input_data=[[input_district,input_season,input_soil_quality,input_ph_4_45,input_ph_45_5,input_ph_5_55,input_ph_55_6,input_ph_6_65,input_ph_65_7,input_ph_7_75]]

    crop = c_values[int(cn2_classifier(input_data))]

    return crop

def ph_fun(pH_value):
    input_ph_4_45 = 0
    input_ph_45_5 = 0
    input_ph_5_55 = 0
    input_ph_55_6 = 0
    input_ph_6_65 = 0
    input_ph_65_7 = 0
    input_ph_7_75 = 0

    if pH_value < 4.5 :
        input_ph_4_45 = 1

    elif pH_value < 5 :
        input_ph_45_5 = 1

    elif pH_value < 5.5 :
        input_ph_5_55 = 1

    elif pH_value < 6 :
        input_ph_55_6 = 1

    elif pH_value < 6.5 :
        input_ph_6_65 = 1

    elif pH_value < 7 :
        input_ph_65_7 = 1

    else :
        input_ph_7_75 = 1

    return input_ph_4_45, input_ph_45_5, input_ph_5_55, input_ph_55_6, input_ph_6_65, input_ph_65_7, input_ph_7_75
