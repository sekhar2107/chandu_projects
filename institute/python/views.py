from django.shortcuts import render
from .models import Student


#def login(req):
#    userdetails = {'username':'chandu2107','pass':'chandu'}
#    return render(req, "login.html",userdetails)
# Create your views here.
def student_table(request):
    
    data = Student.objects.all()
    return render(request, 'students.html', {'students': data})

def employee(req):
    employeedetails = {'Name':'Chandu','ID':'7507','Salary':'90000','City':'Hyderabad'}
    
    return render(req, "employee.html",employeedetails)
   
   