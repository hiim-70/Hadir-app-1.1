import os
import re  # regex

from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from .forms import RegisterForm
from django.core.mail import send_mail, BadHeaderError
from django.shortcuts import render, redirect
from .models import Student, Class, Image, Attendance, Absence, Date, Traning

from datetime import date


import PIL.Image

from LDA_TEST import *
from RecognitionSystem.FaceDetection import *
from RecognitionSystem.FaceRecognition import *

import sys
sys.setrecursionlimit(10000)  # to solve the setrecursionlimit error


def DeleteFolderIfExist(Path):
    for filename in os.listdir(Path):
        file_path = os.path.join(Path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def temp(request):
    return redirect('/Hadir/')


def index(request):
    return render(request, 'HadirApp/index.html')


@login_required(login_url='./login')
def detail(request):
    users = User.objects.all()
    students = Student.objects.all()
    classes = Class.objects.all()
    context = {'users': users, 'students': students, 'classes': classes}
    return render(request, 'HadirApp/detail.html', context)


def registerPage(request):

    if request.user.is_authenticated:
        return redirect('/Hadir/main')
    else:
        if request.method == 'POST':
            form = RegisterForm(request.POST)
            if form.is_valid():
                if User.objects.filter(email=request.POST['email']).exists():

                    err = (f'A user with this Email already exist!')
                    # exist = authenticate(user)
                    return render(request, 'HadirApp/register.html', {'err': err, 'form': form})
                else:
                    form.save()
                    user = form.save()
                    login(request, user)

                    return redirect('/Hadir/main')
                    # return render(request, 'HadirApp/MainPage.html', {'user': user})

            elif request.POST['password1'] != request.POST['password2']:
                err = (f'Passwords Doesnt Match!')
                return render(request, 'HadirApp/register.html', {'err2': err, 'form': form})
            else:
                print(form.errors.as_text())
                err = (form.errors.as_text())
                return render(request, 'HadirApp/register.html', {'err2': err, 'form': form})

        else:
            print(request.method)

            form = RegisterForm()
            user = None
        return render(request, 'HadirApp/register.html', {'form': form})


def loginPage(request):  # redirect to the page user was on

    if request.user.is_authenticated:
        return redirect('/Hadir/main')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            logPassword = request.POST.get('password')
            # rawPass = logPassword.clear

            user = authenticate(request, username=username,
                                password=logPassword)
            if user is not None:
                login(request, user)
                print(f'User {user} has logged in succesfuly')
                # return render(request, 'HadirApp/MainPage.html', {'user': user})
                return redirect('/Hadir/main')
            else:
                Err = (' The Password is Invalid')
                return render(request, './HadirApp/login.html', {'Err': Err})
        else:
            return render(request, './HadirApp/login.html')


@login_required(login_url='./login')
def LogoutUser(request):
    logout(request)
    return redirect('/Hadir/login')


@login_required(login_url='./login')
def mainPage(request):
    return render(request, 'HadirApp/MainPage.html')


@login_required(login_url='/Hadir/Classes/login')
def student_enrollment(request, class_name, class_id):

    if request.method == 'POST':
        name = request.POST['name']
        student_id = request.POST['student_id']
        images = request.FILES.getlist('images')

        try:
            if Class.objects.filter(class_id=class_id).exists:
                classes = Class.objects.get(class_id=class_id)
                print(classes)

                match = re.match(r'^(4)(\d{2}0\d{4})$', student_id)

                if Student.objects.filter(student_id=student_id, name=name).exists():

                    st = Student.objects.get(student_id=student_id, name=name)
                    print(st.classes.all())

                    for clas in st.classes.all():
                        if clas == classes:
                            idErr = 'This Student is already registered in this class'
                            print(idErr)
                            return render(request, 'HadirApp/student_enrollment.html', {'idErr': idErr, 'class_name': class_name})

                    st.classes.add(classes)
                    st.save()

                    messages.success(
                        request, f'Student {st.name} has been added to {classes.class_name} class succesfuly')
                    return render(request, 'HadirApp/student_enrollment.html', {'class_name': classes.class_name})

                elif Student.objects.filter(student_id=student_id).exists():
                    idErr = 'There exists a Student with this ID but with deffernt name.'
                    print(idErr)
                    return render(request, 'HadirApp/student_enrollment.html', {'idErr': idErr, 'class_name': class_name})

                elif not match:
                    wrongID = 'Invalid Student ID!'
                    print(wrongID)
                    return render(request, 'HadirApp/student_enrollment.html', {'wrongID': wrongID, 'class_name': class_name})

                if len(images) < 1:
                    ImgErr = 'Please insert face images'
                    print(ImgErr)
                    return render(request, 'HadirApp/student_enrollment.html', {'Err': ImgErr, 'class_name': class_name})

                ProcessingPath = resource_path('RecognitionSystem/Processing/')
                DeleteFolderIfExist(ProcessingPath)

                n = 1
                for image in images:
                    X = PIL.Image.open(image)
                    X.save(f'{ProcessingPath}/{n}.jpg')
                    n += 1

                std_path = resource_path(
                    f'HadirApp/media/Students/{student_id}/')
                DeleteFolderIfExist(std_path)

                # Detect Submitted Faces
                ImgsNames = DetectFaces(ImgSize=(1024, 1024), ImagePath=ProcessingPath, ConfidenceThreshold=0.87,
                                        std_id=student_id,  Save_noBG=True, Save_cropped=True, pad=30, gain=1.01)

                # If we didn't detect any face -> try with less confidence
                if len(ImgsNames) < 1:
                    ImgsNames = DetectFaces(ImgSize=(800, 800), ImagePath=ProcessingPath,
                                            ConfidenceThreshold=0.8, std_id=student_id, Save_noBG=True, Save_cropped=False)

                # if we got detection
                if len(ImgsNames) >= 1:
                    student = Student.objects.create(
                        name=name, student_id=student_id, profilePic=images[0])
                    student.classes.add(classes)

                    student.save()
                    print(f'student {student} is registered')

                    # Import to Database
                    for im in ImgsNames:
                        Image.objects.create(
                            student=student, images=(f'{im}'))

                    messages.success(
                        request, f'Student {name} has been added to {classes.class_name} class succesfuly')

                else:
                    print('Didnt detect any faces.')
                    Err = 'Didnt detect any faces'
                    return render(request, "HadirApp/student_enrollment.html", {'class_name': class_name, 'Err': Err})
            else:
                print('404 Class Not Found')
                return redirect('/Hadir/404')

        except Exception as e:
            print(e)
            return redirect('/Hadir/404')

    return render(request, 'HadirApp/student_enrollment.html', {'class_name': class_name})


# to show all images // (insignificant)


@login_required(login_url='./login')
def images(request):
    image = Image.objects.all()
    for img in image:
        print(img)
    return render(request, 'HadirApp/images.html', {'image': image})


@login_required(login_url='./login')
def create_class(request):
    if request.method == 'POST':
        classID = request.POST['classID']
        className = request.POST['className']
        numOfStudents = request.POST['numOfStudents']

        matchName = re.match(r'\D{3,50}', className)
        matchID = re.match(r'\d{3}', classID)

        if not matchName:
            nameErr = 'Invalid Class Name'
            return render(request, 'HadirApp/class_form.html', {'nameErr': nameErr})
        elif not matchID:
            IDErr = 'Invalid Class ID'
            return render(request, 'HadirApp/class_form.html', {'IDErr': IDErr})

        try:
            if Class.objects.filter(class_id=classID).exists():
                idErr = 'A Class with this ID already exsists'
                print(idErr)
                return render(request, 'HadirApp/class_form.html', {'idErr': idErr})

            newClass = Class.objects.create(
                class_id=classID, class_name=className, num_of_students=numOfStudents, instructor=request.user)
            newClass.save()
            succes = (f"Class {className}-{classID} Created Succesfuly")
            print(succes)
            return redirect(f'/Hadir/student_enrollment/{className}-{classID}')
        except:
            print('404 Not Found')
            return redirect('/Hadir/404')
    return render(request, 'HadirApp/class_form.html')


@login_required(login_url='./login')
def Classes(request):
    if Class.objects.filter(instructor=request.user).exists():
        theClasses = []
        try:
            for clas in Class.objects.filter(instructor=request.user):
                if clas.instructor == request.user:
                    theClasses.append(clas)
                    print(clas)
        except:
            if Class.objects.filter(instructor=request.user).instructor == request.user:
                theClasses.append(clas)
        return render(request, 'HadirApp/classes.html', {'theClasses': theClasses})

    else:
        err = "You dont Have Any Classes"
        return render(request, 'HadirApp/classes.html', {'err': err})


@login_required(login_url='/Hadir/login?next=/Hadir/Classes')
def deleteClass(request, class_id, class_name):

    try:
        if Class.objects.filter(instructor=request.user).exists():
            DeletedClas = Class.objects.get(class_id=class_id).delete()
            print(f"class: {DeletedClas} Has been deleted succefuly")

            if Class.objects.filter(instructor=request.user).exists():
                theClasses = []
                try:
                    for clas in Class.objects.filter(instructor=request.user):
                        if clas.instructor == request.user:
                            theClasses.append(clas)
                            print(clas)
                except:
                    if Class.objects.filter(instructor=request.user).instructor == request.user:
                        theClasses.append(clas)

                return render(request, 'HadirApp/classes.html', {'theClasses': theClasses, 'Delclass': DeletedClas})
        else:

            err = "You dont Have Any Classes"
            return render(request, 'HadirApp/classes.html', {'err': err, 'Delclass': DeletedClas})

    except Exception as e:
        print(e)
        return redirect('/Hadir/404')


@login_required(login_url='/Hadir/login?next=/Hadir/Classes')
def deleteStudent(request, class_name, class_id, student_id, name):

    if Class.objects.filter(instructor=request.user).exists():

        try:
            clas = Class.objects.get(class_id=class_id)
            students = Student.objects.filter(classes=clas)
            student = Student.objects.get(student_id=student_id)
            student.classes.remove(clas)
            print(f"Student: {student} Has been deleted from {clas} succefuly")

            return render(request, "HadirApp/dashboard.html", {'class_name': class_name, 'class_id': class_id, 'delstudent': student, 'students': students})
        except Exception as e:
            print(e)
            return redirect('/Hadir/404')
    else:
        return redirect('/Hadir/404')


@login_required(login_url='/Hadir/login?next=/Hadir/Classes')
def clas(request, class_id, class_name):
    try:
        if Class.objects.filter(class_id=class_id).exists():
            currentClass = Class.objects.get(class_id=class_id)
            students = Student.objects.all()
            classStd = []
            for student in students:
                for clas in student.classes.all():
                    if clas == currentClass:
                        classStd.append(student)
                        # classStd.save()

            if not classStd:
                print('no student in this class')

            print(classStd)

            return render(request, 'HadirApp/class.html', {'classStd': classStd, 'currentClass': currentClass})
        else:
            print('class 404 Not Found')
            return redirect('/Hadir/404')
            # here
    except:
        print('404 Not Found')
        return redirect('/Hadir/404')

    return render(request, 'HadirApp/class.html')


@login_required(login_url='/Hadir/login?next=/Hadir/Classes')
def attendance(request, class_name, class_id):

    if Class.objects.filter(class_id=class_id, instructor=request.user).exists():

        today = date.today()
        currentClass = Class.objects.get(class_id=class_id)
        students = Student.objects.filter(classes=currentClass)

        if request.method == "POST":

            FolderToSave = resource_path('RecognitionSystem/Processing/')
            TakenImages = request.FILES.getlist('Images')
            DeleteFolderIfExist(FolderToSave)

            prestudents = []

            if len(TakenImages) > 0:
                n = 0
                for image in TakenImages:
                    X = PIL.Image.open(image)
                    X.save(f'{FolderToSave}{n}.jpg')
                    n += 1

                DetectFaces(ImgSize=(1024, 1024), ImagePath=FolderToSave, ConfidenceThreshold=0.8,
                            Save=False,  Save_noBG=False, Save_cropped=True,  pad=30, gain=1.01)

                print(
                    f'Type of Students {type(students)} and its values are {students}')
                students_ids = Recognize_LDA(students)

                for id in students_ids:
                    if Student.objects.filter(student_id=id, classes=currentClass).exists():
                        prestudents.append(Student.objects.get(student_id=id))

            else:
                studentsNames = request.POST.getlist('student')
                for name in studentsNames:
                    prestudents.append(Student.objects.get(name=name))

            print('')
            print(f'Date: {today}')
            print("------------------------------")
            print(f'class: {currentClass} ')
            print("------------------------------")

            if Attendance.objects.filter(presence_date=today, clas=currentClass).exists():
                print("Exist (Attendance is Already took)")

                day = Attendance.objects.get(
                    presence_date=today, clas=currentClass)
                print(f'Attandance for: {day}')
                for st in prestudents:
                    day.student.add(st)
                    day.save()
                    print(f'{st} Marked As Present!')

                abcentStudents = [
                    student for student in students if student not in prestudents]

                print("------------------------------")
                for student in abcentStudents:

                    if Date.objects.filter(date=today).exists():
                        DATE = Date.objects.get(date=today)
                        DATE.save()

                        pass
                    else:
                        DATE = Date.objects.create(date=today)
                        DATE.save()

                    if Absence.objects.filter(student=student, clas=currentClass).exists():
                        DATE = Date.objects.get(date=today)
                        DATE.save()
                        if Absence.objects.filter(student=student, clas=currentClass, date=DATE).exists():

                            print(
                                f"student {student} is already marked absent")
                            pass

                        else:
                            absent = Absence.objects.get(
                                student=student, clas=currentClass)
                            print(absent.counter)
                            absent.counter = absent.counter + 1
                            print(absent.counter)
                            # absent.save()
                            absent.date.add(DATE)
                            absent.save()
                            print(DATE)
                            print(f" Student {student} is Abcent")
                    else:
                        absent = Absence.objects.create(
                            student=student, clas=currentClass)
                        absent.counter = absent.counter + 1
                        absent.date.add(DATE)
                        absent.save()
                        print(f" Student {student} is Abcent")
                return redirect('./Results')

            else:
                day = Attendance.objects.create(
                    presence_date=today, clas=currentClass)
                day.save()
                print(f'{day} CREATED!')

                for st in prestudents:

                    day.student.add(st)
                    day.save()
                    print(f'{st} Marked As Present!')

                abcentStudents = [
                    student for student in students if student not in prestudents]

                for student in abcentStudents:

                    if Date.objects.filter(date=today).exists():
                        DATE = Date.objects.get(date=today)
                        DATE.save()

                        pass
                    else:
                        DATE = Date.objects.create(date=today)
                        DATE.save()

                    print(f" Student {student} is Abcent")
                    if Absence.objects.filter(student=student, clas=currentClass).exists():
                        absent = Absence.objects.get(
                            student=student, clas=currentClass)
                        print(absent.counter)
                        absent.counter = absent.counter + 1
                        absent.date.add(DATE)
                        print(absent.counter)
                        absent.save()
                    else:
                        absent = Absence.objects.create(
                            student=student, clas=currentClass)
                        absent.counter = absent.counter + 1
                        absent.date.add(DATE)
                        absent.save()
                return redirect('./Results')

        context = {'students': students,
                   'class_name': class_name, 'class_id': class_id}
        return render(request, 'HadirApp/attendance.html', context)
    else:
        return redirect('/Hadir/404')


@login_required(login_url='/Hadir/login?next=/Hadir/Classes')
def attendanceResult(request, class_name, class_id):

    try:

        today = date.today()

        # to get all present students today
        day = Attendance.objects.filter(presence_date=today)
        clas = Class.objects.get(class_id=class_id)
        students = Student.objects.filter(classes=clas)
        for st in day:
            prestudents = st.student.all()
        # print(f' students: {prestudents}')

        currentClass = Class.objects.get(class_id=class_id)
        students = Student.objects.filter(
            classes=currentClass)
        abcentStudents = [
            student for student in students if student not in prestudents]
        today = str(today).replace(' ', "-")
    except Exception as e:
        print(e)
        return redirect('/Hadir/404')
    return render(request, 'HadirApp/results.html', {'currentClass': currentClass, 'today': today, 'prestudents': prestudents, 'abcentStudents': abcentStudents, 'students': students})


@login_required(login_url='/Hadir/login?next=/Hadir/Classes')
def dashboard(request, class_name, class_id):

    if Class.objects.filter(class_id=class_id, instructor=request.user).exists():
        clas = Class.objects.get(class_id=class_id, instructor=request.user)
        students = Student.objects.filter(classes=clas)
        Ablist = []
        if len(students) > 0:
            for student in students:
                absence = Absence.objects.filter(student=student, clas=clas)
            # Ablist.append(len(absence))

                return render(request, 'HadirApp/dashboard.html', {'class_name': class_name, 'class_id': class_id, 'students': students, 'absence': absence})
        else:

            return render(request, 'HadirApp/dashboard.html', {'class_name': class_name, 'class_id': class_id, 'students': students, })

    else:
        return redirect('/Hadir/404')


def profile(request, class_name, class_id, student_id, name):

    student = Student.objects.get(student_id=student_id)
    clas = Class.objects.get(class_id=class_id)
    # print(student.name)
    if Absence.objects.filter(student=student, clas=clas).exists():
        absence = Absence.objects.get(student=student, clas=clas)
        persintage = str(((absence.counter/24)*100))[:3]
    else:
        absence = 0
        persintage = 0
    context = {
        'persintage': persintage, 'absence': absence, 'class': clas, 'student': student
    }
    return render(request, 'HadirApp/profile.html', context)


@login_required(login_url='./login')
def traning(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')

        FolderToSave_Training = resource_path('HadirApp/media/Students')
        FolderToSave_Processing = resource_path(
            'RecognitionSystem/Processing/')

        DeleteFolderIfExist(FolderToSave_Processing)

        if os.path.exists(f'{FolderToSave_Training}/99999999') is False:
            os.makedirs(f'{FolderToSave_Training}/99999999')
        else:
            DeleteFolderIfExist(f'{FolderToSave_Training}/99999999')
            # os.makedirs(f'{FolderToSave_Training}/99999999')

        for img in images:
            i = PIL.Image.open(img)
            i.save(
                f'{FolderToSave_Processing}{len(os.listdir(FolderToSave_Processing))}.png')

        n = len(os.listdir(FolderToSave_Training))
        DetectionFilesName = DetectFaces(ImgSize=(96, 96), ImagePath=FolderToSave_Processing, ConfidenceThreshold=0.8,
                                         MediaPath=FolderToSave_Training, std_id='99999999',  Save_noBG=False, Save_cropped=True, pad=30, gain=1)

        print('Traning set added succesfuly')
    return render(request, 'HadirApp/Traning.html')


def PageNotFound(request):

    return render(request, 'HadirApp/404.html')


def handler404(request, exception):
    return render(request, 'HadirApp/404.html')


def handler500(request):
    return render(request, 'HadirApp/404.html')
