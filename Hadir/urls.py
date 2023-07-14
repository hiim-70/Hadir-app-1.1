from django.contrib import admin
from django.urls import re_path, path, include
from HadirApp import views  # . = current folder
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.views.static import serve
urlpatterns = [


    re_path(r'^media/(?P<path>.*)$', serve,
            {'document_root': settings.MEDIA_ROOT}),

    re_path(r'^static/(?P<path>.*)$', serve,
            {'document_root': settings.STATIC_ROOT}),


    path('admin/', admin.site.urls),
    # http://localhost:8000/admin/

    path('', views.temp, name="temp"),
    # http://localhost:8000/Hadir/


    path('Hadir/', views.index, name="index"),
    # http://localhost:8000/Hadir/

    path('Hadir/detail', views.detail, name="detail"),
    # http://localhost:8000/Hadir/detail


    path('Hadir/student_enrollment/<str:class_name>-<int:class_id>',
         views.student_enrollment, name="student_enrollment"),
    # http://localhost:8000/Hadir/student_enrollment

    path('Hadir/Create_class', views.create_class, name="create_class"),
    # http://localhost:8000/Hadir/Create_class


    path('Hadir/Classes', views.Classes, name="Classes"),
    # http://localhost:8000/Hadir/Classes


    path('Hadir/Classes/<str:class_name>-<int:class_id>', views.clas, name="clas"),
    # http://localhost:8000/Hadir/math-103

    path('Hadir/Classes/<str:class_name>-<int:class_id>/Delete',
         views.deleteClass, name="deleteClass"),
    # http://localhost:8000/Hadir/math-103/delete

    path('Hadir/Classes/<str:class_name>-<int:class_id>/<str:name>-<int:student_id>/Delete',
         views.deleteStudent, name="deleteStudent"),
    # http://localhost:8000/Hadir/math-103/khalid-43901023/delete


    path('Hadir/Classes/<str:class_name>-<int:class_id>/Dashboard',
         views.dashboard, name="dashboard"),
    # http://localhost:8000/Hadir/Dashboard

    path('Hadir/Classes/<str:class_name>-<int:class_id>/<str:name>-<int:student_id>',
         views.profile, name="profile"),
    # http://localhost:8000/Hadir/Marwan%20Saleh%20Algamdi


    path('Hadir/Classes/<str:class_name>-<int:class_id>/Attendance',
         views.attendance, name="attendance"),
    # http://localhost:8000/Hadir/Attendance


    path('Hadir/Classes/<str:class_name>-<int:class_id>/Results',
         views.attendanceResult, name="attendanceResult"),
    # http://localhost:8000/Hadir/--


    #     path('Hadir/images', views.images, name='images'),
    # http://localhost:8000/Hadir/images

    path('Hadir/Traning', views.traning, name='traning'),
    # http://localhost:8000/Hadir/Traning


    path('Hadir/register', views.registerPage, name="registerPage"),
    # http://localhost:8000/Hadir/register


    path('Hadir/login', views.loginPage, name="loginPage"),
    # http://localhost:8000/Hadir/login


    path('Hadir/logout', views.LogoutUser, name="LogoutUser"),


    path('Hadir/reset_password/',
         auth_views.PasswordResetView.as_view(template_name='HadirApp/password_reset.html'), name="reset_password"),

    path('Hadir/reset_password_sent/',
         auth_views.PasswordResetDoneView.as_view(template_name='HadirApp/password_reset_sent.html'), name="password_reset_done"),

    path('Hadir/reset_password/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(template_name='HadirApp/reset_password.html'), name="password_reset_confirm"),

    path('Hadir/reset_password_complete/',
         auth_views.PasswordResetCompleteView.as_view(template_name='HadirApp/reset_password_complete.html'), name="password_reset_complete"),


    path('Hadir/main', views.mainPage, name="mainPage"),
    # http://localhost:8000/Hadir/main


    path('Hadir/404', views.PageNotFound, name="PageNotFound")
    # http://localhost:8000/Hadir/404

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# X:\TU\Capstone project\Engine\Django\Project\Hadir\HadirApp\views.py
# X:\TU\Capstone project\Engine\Django\Project\Hadir\Hadir\urls.py


handler404 = 'HadirApp.views.handler404'
handler500 = 'HadirApp.views.handler500'
