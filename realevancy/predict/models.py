from django.db import models

# Create your models here.

class PredResults(models.Model):

    keyword = models.CharField(max_length=50)
    link = models.CharField
    classification = models.CharField(max_length=30)

    def __str__(self):
        return self.classification