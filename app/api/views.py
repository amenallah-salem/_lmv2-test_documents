from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.conf import settings
import time
import os 
import json
from datetime import datetime, date
from django.http import HttpResponse 
from django.shortcuts import render
from tesserocr import PyTessBaseAPI
import imghdr
from .models import pdfFile
import os # folder directory navigation
from PIL import Image, ImageFilter
from tesserocr import PyTessBaseAPI
import PIL
from ._preprocess import extract_txt_from_pdf
import os
from .serialiszer import pdf_serializer
##PIL to Base64
import base64
from io import BytesIO
def save_load(_name_th_, image_obj_):
    save_dir = f"{_name_th_.split('.')[-2]}_pred.{_name_th_.split('.')[-1]}"
    image_obj_.save(save_dir)
    try:
        go= pdfFile.objects.create(pdfFile=save_dir)
        inference_im_name = save_dir.split('pdf/')[-1]
    except pdfFile.DoesNotExist:
        go = None
    MAIN_DIR = "http://0.0.0.0:8000/" #os.getcwd()
    inference_image_path="./media/"+save_dir
    print('infer name ', inference_image_path)
    return inference_image_path

def pil_base64(image):
  img_buffer = BytesIO()
  image.save(img_buffer, format='JPEG')
  byte_data = img_buffer.getvalue()
  base64_str = base64.b64encode(byte_data)
  return base64_str
 
 
##Base64 to pil
import base64
from io import BytesIO
from PIL import Image
 
def base64_pil(base64_str):
  image = base64.b64decode(base64_str)
  image = BytesIO(image)
  image = Image.open(image)
  return image

# os.system('pip install pyyaml==5.1')
# # workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
# os.system('pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')

# # install detectron2 that matches pytorch 1.8
# # See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# os.system('pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html')

# ## install PyTesseract
# os.system('pip install -q pytesseract')

import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
def normalize_box(bbox,width,height):
  return [
          int(bbox[0]*(1000/width)),
          int(bbox[1]*(1000/height)),
          int(bbox[2]*(1000/width)),
          int(bbox[3]*(1000/height)),
  ]
def get_json(image_path, img_obj, actual_boxes, predictions):
    im_rel_path=image_path.split('.')[-2]
    im_name=im_rel_path.split('/')[-1]
    tsv_out_path= f"./tsv_output-{im_name}"
    os.system(f"tesseract '{image_path}' '{tsv_out_path}' -l eng tsv")
    ocr_df = pd.read_csv(f"{tsv_out_path}.tsv", sep='\t')
    ocr_df = ocr_df.dropna()
    ocr_df = ocr_df.drop(ocr_df[ocr_df.text.str.strip() == ''].index)
    text_output = ocr_df.text.tolist()
    doc_text = ' '.join(text_output)
    width, height = img_obj.size
    words = []
    for index,row in ocr_df.iterrows():
        word = {}
        origin_box = [row['left'],row['top'],row['left']+row['width'],row['top']+row['height']] 
        word['word_text'] = row['text']
        word['word_box'] = origin_box
        word['normalized_box'] = normalize_box(word['word_box'],width, height)
        words.append(word)
    boxlist = [word['normalized_box'] for word in words]
    wordlist = [word['word_text'] for word in words]
    for word in words:
        word_labels = [] 
        token_labels = []
        word_tagging = None 
        for i,box in enumerate(actual_boxes,start=0):
            word_labels.append(predictions[i])
            token_labels.append(predictions[i])
            if word_labels != []:
                word_tagging =  word_labels[i] 
            word['word_labels'] = token_labels
            word['word_tagging'] = word_tagging
        filtered_words = [{'id':i,'text':word['word_text'],
                        'label':predictions[i],
                        'box':word['word_box'],
                        'words':[
                            {'box':word['word_box'],
                             'text':word['word_text']}
                        ]
                    } for i,word in enumerate(words)]
    return {"out":filtered_words}
    
    
 

# #------------padel dependencies------------------------
# #define the ocr engine for PPADEL OCR 
# from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
# from matplotlib import pyplot as plt # plot images
# import cv2 #opencv
# ocr_model = PaddleOCR(ocr = PaddleOCR(use_angle_cls=True, lang='en'))
#------------EXTRACT INVOICE------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""STATIC_CONFIG FOR MODELS"""
def ex_time(method):
    """excution time decorator
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('#TIME in %r is %2.3f ms' % (method.__name__, (te - ts) * 1000))
        return result    
    return timed
def file_type(filename):
    print(filename.split(".")[-1])
    if filename.split(".")[-1] in ["png", "jpeg","jpg"]:
        return "image"
    elif filename.split(".")[-1] in ["pdf"]:
        return "pdf"
    else:
        return "UNSUPPORTED file type"

#-------------simple_invoice_extraction----------------------------------------
#all defined outside the api to not reload exery time 
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")



model_simple_invoice = LayoutLMv2ForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv2-finetuned-sroie_mod")
dataset = load_dataset("darentang/generated", split="test")
labels = dataset.features['ner_tags'].feature.names
print('labels', labels)
id2label = {v: k for v, k in enumerate(labels)}
label2color = {
    'b-abn': "blue",
    'b-biller': "blue",
    'b-biller_address': "black",
    'b-biller_post_code': "green",
    'b-due_date': "orange",
    'b-gst': 'red',
    'b-invoice_date': 'red',
    'b-invoice_number': 'violet',
    'b-subtotal': 'green',
    'b-total': 'green',
    'i-biller_address': 'blue',
    'o': 'violet'
 }
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    return label


def process_image_invoice(im_path):
    """Input file path :str()
    Return a PIL.Image.Image object
    """
    image=Image.open(im_path).convert("RGB")
    width, height = image.size
    # encode
    encoding = processor(
        image, 
        truncation=True, 
        return_offsets_mapping=True, 
        return_tensors="pt"
    )
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model_simple_invoice(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    json_resp=get_json( image_path = im_path, img_obj = image, actual_boxes=true_boxes, predictions=true_predictions)

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()# to change to clé valeur 
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)
    return image, json_resp
    
#np.array(image).tolist(), json_resp


#-------------simple_receipe_extraction----------------------------------------


model_recepies=LayoutLMv2ForTokenClassification.from_pretrained("katanaml/layoutlmv2-finetuned-cord")
id2label_2 = model_recepies.config.id2label
label_ints = np.random.randint(0,len(PIL.ImageColor.colormap.items()),30)
label_color_pil = [k for k,_ in PIL.ImageColor.colormap.items()]
label_color = [label_color_pil[i] for i in label_ints]
label2color_2 = {}
for k,v in id2label_2.items():
  if v[2:] == '':
    label2color_2['o']=label_color[k]
  else:
    label2color_2[v[2:]]=label_color[k]

def iob_to_label_2(label):
    label = label[2:]
    if not label:
        return 'o'
    return label

def process_image_2(image_path):
    image=Image.open(image_path).convert("RGB")
    width, height = image.size

    # encode
    encoding = processor(image, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model_recepies(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label_2[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    json_resp=get_json(image_path = image_path, img_obj = image, actual_boxes=true_boxes, predictions=true_predictions)
    print(json_resp)
    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label_2(prediction).lower()
        draw.rectangle(box, outline=label2color_2[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color_2[predicted_label], font=font)

    return image, json_resp
    
#np.array(image).tolist()
# #------------question answer code  dependencies------------------------

model_qa = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")

# load image example
dataset_qa = load_dataset("nielsr/funsd", split="test")
# define id2label, label2color
labels_qa = dataset_qa.features['ner_tags'].feature.names
id2label_qa_1 = {v: k for v, k in enumerate(labels_qa)}

label2color_qa_1 = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

def iob_to_label_qa(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

def process_image_o(image_path):
    image=Image.open(image_path).convert("RGB")
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model_qa(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label_qa_1[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    json_resp=get_json( image_path = image_path, img_obj = image, actual_boxes=true_boxes, predictions=true_predictions)
    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label_qa(prediction).lower()
        draw.rectangle(box, outline=label2color_qa_1[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color_qa_1[predicted_label], font=font)
    
    return image, json_resp
    

#------------vues------------------------


@api_view(['GET'])
def test(request):
        try:
            return JsonResponse({"succ": True, "msg": "API is working correctly"})
        except Exception as e :
            return JsonResponse({
                "succ": False , "msg": "the api is under maintainance, please contact the author"
            })

@api_view(["POST"])
def extract(request):
    """
    supported jobs: 
    _extract_ocr :         Return : PURE TEXT                                               DONNE
    _preprocess            Return: Json of preprocessed doc/image                           NO
    _clean                 Return: clean image                                              NO
    _extract_invoice       Json & img                                                       DONNE
    _extract_recipes       Json & img                                                       DONNE
    _legal_documents       Json & img
    _question_answer       Json & img
    _NGP                   Json & img
    """
    if request.method=="POST":
        #-----------GENERAL PIPELINE-----------------------------------------------------------------------
        try:
            required_job = request.data['job']
            _file = request.data['file']
        except Exception as e:
            return JsonResponse({"succ": False, "err_msg": "job or pdf/image is missing"})
        _doc = pdfFile.objects.create(pdfFile=request.FILES['file'])
        _doc.save()
        _name_th=os.path.join(settings.MEDIA_ROOT,str(_doc.pdfFile.name))
        print("file_type ::",file_type(_name_th))
        #-----------PIPELINE:: _extract_ocr---------------------------------------------------------------
        if required_job == "_extract_ocr":
            if file_type(_name_th) == "image": # if the file is an image
                with PyTessBaseAPI() as api:
                    with Image.open(_name_th) as image:
                        sharpened_image = image.filter(ImageFilter.SHARPEN)#image_filter to clarify the image
                        api.SetImage(sharpened_image)#to get full text
                        utf8_text = api.GetUTF8Text()
                return JsonResponse({"succ": True, "resp": utf8_text})
            elif file_type(_name_th)=='pdf':
                return JsonResponse({"succ": False, "resp": extract_txt_from_pdf(_name_th)})
            else: return JsonResponse({"succ": False, "resp": "prob:: please verify your input"})
        #-----------PIPELINE:: _preprocess---------------------------------------------------------------
        elif required_job == "_analyse":
            try: 
                return  JsonResponse({
                    "succ": True , "resp" :extract_txt_from_pdf(_name_th) }) 
            except:
                return  JsonResponse({
                    "succ": False , "resp": "error was occured"
                })  


        #-----------PIPELINE:: _clean---------------------------------------------------------------
        # elif required_job == "_clean": # https://github.com/nicknochnack/DrugLabelExtraction- (paddleOCR)
        #     if file_type == "image":
        #         try:
        #             ocr_model = PaddleOCR(lang='en')
        #         except Exception as e :
        #             return JsonResponse({"succ": False, "err_msg": "this feature requires GPU support wich is not taken in consideration for this machine:: please wait for the next release"})
        #         result = ocr_model(_name_th)
        #         if result:
        #             return JsonResponse({"succ": True, "resp": result})
        #         else:
        #             return JsonResponse({"succ": False, "resp": "an error was occured during extraction, please change the image or reload the page"})
        #-----------PIPELINE:: _extract_invoice---------------------------------------------------------------

    
        elif required_job== "_extract_invoice":

            """this is a simple invoice data extraction using layoutlmv2
            link: https://huggingface.co/spaces/Theivaprakasham/layoutlmv2_invoice/blob/main/app.py
            """
            try:
                image_obj_inv, res=process_image_invoice(_name_th)
                inference_image= save_load(_name_th, image_obj_inv)
                return  JsonResponse({
                    "succ": True, "inference_image":inference_image , "resp": res 
                })
            
            except Exception as e:
                return JsonResponse({"succ": False, "err_msg": str(e)})



        elif required_job=="_extract_recipes":
            """this is a simple receipe data extraction using layoutlmv2
            link: https://huggingface.co/spaces/katanaml/LayoutLMv2-CORD/blob/main/app.py
            https://huggingface.co/spaces/katanaml/LayoutLMv2-CORD
            """
            try:
                image_obj_res, res=process_image_2(_name_th)
                inference_image= save_load(_name_th, image_obj_res)
                return  JsonResponse({
                    "succ": True, "inference_image":inference_image , "resp": res 
                })
            except Exception as e:
                return JsonResponse({"succ": False, "err_msg": str(e)})


        elif required_job=="_legal_documents":
            pass 
        elif required_job== "_question_answer":
            try:
                image_obj_rqa, res=process_image_o(_name_th)
                inference_image= save_load(_name_th, image_obj_rqa)
                return  JsonResponse({
                    "succ": True, "inference_image":inference_image , "resp": res 
                })
            except Exception as e:
                return JsonResponse({"succ": False, "err_msg": str(e)})
        elif required_job== "_NGP":
            return JsonResponse({
                "succ": False, "err_msg": "cette partie  n'est pas stable et est en cours de developement"
            })









