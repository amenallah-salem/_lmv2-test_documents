from django.urls import path
from . import views
urlpatterns = [
    path("extract/", views.extract, name='fast & accurate ocr text & data extraction from any document'),
    path("test/", views.test, name='home page'),
]