from django.shortcuts import render,redirect
from .models import Student
    


# Create your views here.
def login(req):
    return render(req,'login.html')

def register(req):
    
    if req.method == "POST":
        name = req.POST.get("name")
        email = req.POST.get("email")
        password = req.POST.get("password")
        confirmpassword = req.POST.get("confirmpassword")
        
        Student.objects.create(name=name,email=email,password=password,confirmpassword=confirmpassword)
   
        return redirect("sucess",id=Student.id)
    return render(req,'register.html')
# Create your views here.