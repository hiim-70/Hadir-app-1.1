# Generated by Django 3.2.5 on 2023-01-09 11:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('HadirApp', '0006_absence'),
    ]

    operations = [
        migrations.AlterField(
            model_name='student',
            name='name',
            field=models.CharField(max_length=60, unique=True),
        ),
    ]
