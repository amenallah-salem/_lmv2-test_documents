from attr import field
from rest_framework import serializers
from .models import pdfFile
class pdf_serializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = pdfFile
        fields = ["pdfFile"]
        
