from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict',views.predict,name='predict'),
    path('npk_train_model',views.npk_train_model,name='npk_train_model'),
    path('crop_train_model',views.crop_train_model,name='crop_train_model')
]
