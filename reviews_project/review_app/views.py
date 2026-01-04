from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings


from .model import extract_reviews, summarize_reviews


def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "").strip()

        # EMPTY FIELD CHECK
        if username == "" or password == "" or email == "":
            messages.error(request, "All fields are required.")
            return render(request, "register.html")

        # CHECK USERNAME EXISTS
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists! Try another.")
            return render(request, "register.html")

        # CHECK EMAIL EXISTS
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return render(request, "register.html")

        # CREATE USER
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        messages.success(request, "Registration successful! Please log in.")
        return redirect("login")

    return render(request, "register.html")




def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(username=username, password=password)
        if user:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid login")
            return render(request, "login.html")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


def home_view(request):
    if not request.user.is_authenticated:
        return redirect("login")

    if request.method == "POST":
        url = request.POST.get("url")

        extracted = extract_reviews(url)
        if not extracted:
            messages.error(request, "Unable to extract reviews from this URL.")
            return render(request, "home.html")

        summary = summarize_reviews(extracted)

        return render(request, "summary.html", {
            "url": url,
            "summary": summary,
            "extracted": extracted,
        })

    return render(request, "home.html")
