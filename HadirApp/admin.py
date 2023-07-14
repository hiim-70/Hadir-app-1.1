from django.contrib import admin

# Register your models here.

from .models import Student, Class, Image, Attendance, Absence, Traning, Date

admin.site.register(Student)
admin.site.register(Class)
admin.site.register(Image)
admin.site.register(Attendance)
admin.site.register(Absence)
admin.site.register(Traning)
admin.site.register(Date)
