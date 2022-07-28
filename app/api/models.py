
from django.db import models

class pdfFile(models.Model):
    """"pdf model ; works also with images"""
    pdfFile = models.FileField(upload_to='pdf')

    def delete(self, *args, **kwargs):
        self.pdfFile.delete()
        super().delete(*args, **kwargs)

    def __str__(self):
        return str(self.pdfFile.name)

    def save(self, *args, **kwargs):
        super(pdfFile, self).save(*args, **kwargs)