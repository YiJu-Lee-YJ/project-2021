from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from pandas.io import pickle
from .forms import SignUpForm
from django.http import JsonResponse
from django.views.generic import TemplateView
from django.http import HttpResponseRedirect
import pandas as pd
import dill
from tinydb import TinyDB, Query
# Create your views here.


def homepage(request):
    return render(request, 'homepage.html')

def predict(request):
    keyword= request.POST.get('keyword')
    link= request.POST.get('link')
    #replace the path with the my_model in your local machine
    with open(r"/Users/yijulee/Document/FIT3164/model/my_model_final","rb") as f:
        prediction = dill.load(f)
    
    db = TinyDB('./db.json')
    Term = Query()
    temp_dict = db.search( Term.keyword == keyword)[0]
    new_count = temp_dict['count'] + 1
    db.update({'count': new_count}, Term.keyword == keyword)
    data = db.all()

    first,second,third = -1, -1, -1
    kw1, kw2, kw3 = None, None, None

    for keyword2 in data:
        if keyword2['count'] > first:
            first = keyword2['count']
            kw1 = (keyword2['keyword'],keyword2['count'])
        elif keyword2['count'] > second:
            second = keyword2['count']
            kw2 = (keyword2['keyword'],keyword2['count'])
        elif keyword2['count'] > third:
            third = keyword2['count']
            kw3 = (keyword2['keyword'],keyword2['count'])
            
    top_queries = [kw1,kw2,kw3]
    


    res = prediction(keyword,link)
    res[9].savefig('static/image/piechart.png',bbox_inches='tight')
    
    context= {'keyword': keyword, 'link':link, 'title':res[0], 'abstract':res[1], 'article_date':res[2],
    'TotalCases':res[3], 'NewCases':res[4], 'TotalDeaths':res[5], 'NewDeaths':res[6], 'TotalRecovered':res[7], 
    'prediction':res[8], 'res': res, 'topQueries':top_queries, 'kw1':kw1[0], 'count1':kw1[1], 'kw2':kw2[0], 
    'count2':kw2[1], 'kw3':kw3[0], 'count3':kw3[1]}

    return render(request, 'predict.html',context)

def login(request):
    return render(request, 'login.html')

# def signup(request):
#     return render(request, 'signup.html')

def contact(request):
    return render(request, 'contact.html')

def about_us(request):
    return render(request, 'about-us.html')

def privacy(request):
    return render(request, 'privacy.html')

def signup(request):
    form = SignUpForm(request.POST)
    if form.is_valid():
        form.save()
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        user = authenticate(username=username, password=password)
        login(request, user)
        return redirect('home')
    context = {
        'form': form
    }
    return render(request, 'signup.html', context)   

