from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class SignUpForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-input',
            'required':'',
            'name':'username',
            'id':'username',
            'type':'text',
            'placeholder':'Create Username',
            'maxlength': '16',
            'minlength': '6',
            })
        self.fields['email'].widget.attrs.update({
            'class': 'form-input',
            'required':'',
            'name':'email',
            'id':'email',
            'type':'email',
            'placeholder':'Enter email',
            })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-input',
            'required':'',
            'name':'password1',
            'id':'password1',
            'type':'password',
            'placeholder':'Password',
            'maxlength':'22', 
            'minlength':'8'
            })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-input',
            'required':'',
            'name':'password2',
            'id':'password2',
            'type':'password',
            'placeholder':'Retype Password',
            'maxlength':'22', 
            'minlength':'8'
            })            
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    
