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
import Orange
import goslate
import time
from sklearn.externals import joblib
import csv

# Create your views here.
from django.http import HttpResponse

def index(request):
    return render(request, 'index.html')

def npk_train_model(request):
    start_time = time.time()
    df=pd.read_csv("npk.csv")

    X = df.iloc[:,:-1]
    Y = df.iloc[:,3]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    joblib.dump(clf, 'npk_model.pkl')
    stop_time = time.time()

    msg = "Soil Quality prediction model is trained. It uses Decision tree classifier."
    time_required = stop_time - start_time

    return render(request, 'model_train.html',{'msg':msg,'time_required':time_required})

def crop_train_model(request):
    start_time = time.time()
    data = Orange.data.Table("crop.tab")

    cn2_learner = Orange.classification.rules.CN2Learner()
    cn2_classifier = cn2_learner(data)

    joblib.dump(cn2_classifier, 'crop_model.pkl')
    stop_time = time.time()

    msg = "Crop prediction model is trained. It uses CN2 classifier."
    time_required = stop_time - start_time

    return render(request, 'model_train.html',{'msg':msg,'time_required':time_required})

class Prediction_class:
    def __init__(self,nitrogen,phosphorus,potassium,district,season,pH_value):
        self.nitrogen = nitrogen
        self.phosphorus = phosphorus
        self.potassium = potassium
        self.district = district
        self.season = season
        self.pH_value = pH_value
        self.soil_quality = ""
        self.crop = ""
        self.advice = "advice will print here"

    def soil_quality_fun(self,nitrogen,phosphorus,potassium):
        clf = joblib.load('npk_model.pkl')

        soil_quality=clf.predict([[nitrogen,phosphorus,potassium]])
        self.soil_quality = soil_quality[0]

        return self.soil_quality

    def crop_predict(self,district,season,pH_value,soil_quality):
        data = Orange.data.Table("crop.tab")

        cn2_classifier = joblib.load('crop_model.pkl')
        
        c_values = data.domain.class_var.values

        input_district = district
        input_season = season
        input_soil_quality = soil_quality
        input_ph_4_45, input_ph_45_5, input_ph_5_55, input_ph_55_6, input_ph_6_65, input_ph_65_7, input_ph_7_75 = self.ph_fun(pH_value)

        input_data=[[input_district,input_season,input_soil_quality,input_ph_4_45,input_ph_45_5,input_ph_5_55,input_ph_55_6,input_ph_6_65,input_ph_65_7,input_ph_7_75]]

        self.crop = c_values[int(cn2_classifier(input_data))]


    def ph_fun(self,pH_value):
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

    def pesticides_fun(self):
        self.pesticides = np.array(pd.read_csv("pesticides/" + self.crop + ".csv"))

    def insecticides_fun(self):
        self.insecticides = np.array(pd.read_csv("insecticides/" + self.crop + ".csv"))

    def translate(self):
        #text = "Soil quality is " + self.soil_quality + ". You should sow "+ self.crop + "."
        gs = goslate.Goslate()
        #translatedtext = gs.translate(text,'mr')
        #self.advice = translatedtext
        #self.advice = text
        text1 = self.soil_quality
        time.sleep(0.2)
        translatedtext1 = gs.translate(text1,'mr')
        time.sleep(0.2)
        text2 = self.crop
        #time.sleep(15)
        translatedtext2 = gs.translate(text2,'mr')
        #time.sleep(15)
        self.advice = "मातीची गुणवत्ता "+translatedtext1+" आहे. आपण "+translatedtext2+" पेरले पाहिजे."
        #time.sleep(15)


#####---------------------------------------------------------------

def predict(request):
    nitrogen = float(request.POST["nitrogen"])
    phosphorus = float(request.POST["phosphorus"])
    potassium = float(request.POST["potassium"])
    
    district = request.POST["district"]
    season = request.POST["season"]
    pH_value = float(request.POST["pH_value"])

    prediction_object = Prediction_class(nitrogen,phosphorus,potassium,district,season,pH_value)

    #n,p,k soil quality prediction
    soil_quality = prediction_object.soil_quality_fun(nitrogen,phosphorus,potassium)

    #crop prediction
    prediction_object.crop_predict(district,season,pH_value,soil_quality)

    #translate
    #time.sleep(15)
    prediction_object.translate()
    #time.sleep(15)

    prediction_object.pesticides_fun()

    prediction_object.insecticides_fun()

    return render(request,'predict.html',{'prediction_object':prediction_object})



