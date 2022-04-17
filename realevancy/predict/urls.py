from django.urls import path
from . import views

app_name = 'predict'

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('predict/', views.predict, name = 'predict'),
    path('login/', views.login, name = 'login'),
    path('signup/', views.signup, name = 'signup'),
    path('contact/', views.contact, name = 'contact'),
    path('about-us/', views.about_us, name = 'about-us'),
    path('privacy/', views.privacy, name = 'privacy'),
]