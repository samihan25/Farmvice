from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse


def index(request):
    return render(request, 'home.html', {'name':'Samihan'})


def add(request):
    num1=int(request.POST["num1"])
    num2=int(request.POST["num2"])
    result = num1 + num2
    return render(request,'result.html', {'result':result})
