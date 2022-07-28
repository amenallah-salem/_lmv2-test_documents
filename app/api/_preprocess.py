

import json
import yaml
from langdetect import detect_langs
import pdfplumber
import pdftotext
import collections
import nltk
from pdf2image import convert_from_path
from pytesseract import image_to_string
import string
import os

import pytesseract
import shutil

def read_json(json_file_path):
    with open(json_file_path) as f :
        data  = json.load(f)
    return data 

def detect_lang(text):
    """return a list of detected lang with score :
    [en:0.9999956459808726]
    """
    return detect_langs(text)
    
def write_json(data, file_path):
    with open(file_path, 'w') as f :
        json.dump(data, f, indent=2, sort_keys=True)

def read_yaml(yaml_file_path):
    with open(yaml_file_path, "r") as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

def write_yaml(yaml_file_path, data):
    with open(yaml_file_path, "w") as stream:
        try:
            yaml.dump(data, stream)
            print(f"successfully write to {yaml_file_path}")
        except yaml.YAMLError as exc:
            print(exc)

def detect_extention(filename):
    return filename.split('.')[-1]

def check_file_content_class(_page):
    """check if the content of a document is a scanned foc or pure text """
    if _page[0].extract_text() == None:
        return "scanned_doc"
    return "text_doc"

def has_duplicate(text_dic):
    return len(text_dic) != len(set(text_dic))

def rm_duplicates(test_dict):
    ''' Check if given list contains any duplicates '''
    temp = {val : key for key, val in test_dict.items()}
    res = {val : key for key, val in temp.items()}
    return res

def convert_pdf_to_img(pdf_file):
    """returns a set of images from a given pdf that contains scanned images"""
    return convert_from_path(pdf_file)

def convert_image_to_text(file):
    """return extracted text from a given image"""
    return image_to_string(file)

def get_text_from_any_pdf_scanned_content(pdf_file):
    """Get text data from a given pdf file path 
    """
    images = convert_pdf_to_img(pdf_file)
    final_output = {}
    for pg, img in enumerate(images):
        final_output[pg]=convert_image_to_text(img)
    return final_output

def remove_punctuation(text):
    return text.strip(string.punctuation)

def _remove_stopwords(text):
    try:
        stopwords = nltk.corpus.stopwords.words('english')
        output="".join([i for i in text if i not in stopwords])
        return output
    except Exception as e:
        nltk.download('stopwords')
        output = _remove_stopwords(text)

def text_preprocess(txt):
    """
    preprocesses extracted text from document : 
    Input: str object 
    Return : preprocessed text with the following operation made 
    Remove multiple line skips 
    Removing punctuations like . , ! $( ) * % @
    Removing URLs --not sure to do that 
    Removing Stop words
    Lower casing the text 

    """
    try:

        #txt = _remove_punctuation(txt)
        txt = txt.lower()
        #txt = _remove_stopwords(txt)
        txt = txt.replace('\n\n', '\n')
        return txt
    except Exception as e:
        print(e)
        return txt 



def extract_txt_from_pdf(
    path, 
    write_out_txt=False, 
    output_txt_filename="",
    file_type=None
    ):# this will be converted only for buills /contracts has annother layout logic 

    """Inputs:
        - path: str = pdf file relative/abselute path, 
        - write_out_txt: bool = flag if we should save .txt extracted text 
        - output_txt_filename: where to store output txt file,
        - file_type: Bill/contract.
        Returns: 
        dict:{
            file_ext: pdf, Img(jpeg, png ...),
            file_type : bill/contract, to be added later auto detection in v2
            file_content_class: str = text_doc, scanned_doc
            doc_dict: text inside pdf : Dict[page_id: txt]
            doc_text: preprocessed txt
            lang:en/fr...[score]
        }
    """

    file_ext = detect_extention(path)
    file_type = file_type
    text = {}
    
    file_content_class = "" # 
    contains_dupl = False
    removed_dupl = False 

    if file_ext == 'pdf':
        if check_file_content_class(
            pdfplumber.open(path).pages)== "text_doc":
            file_content_class= "pdf document with pure text"
            with pdfplumber.open(path) as pdf:
                pages = pdf.pages
                for page_idx in range(len(pages)):
                    _text  = pages[page_idx].extract_text()
                    text[page_idx] =  _text

        elif check_file_content_class(
            pdfplumber.open(path).pages)== "scanned_doc":
            file_content_class= "pdf document with scanned images"
            text =  get_text_from_any_pdf_scanned_content(path)

        for val_txt in text.values():
            val_txt = text_preprocess(val_txt)
        
        if has_duplicate(text):
            contains_dupl = True
            text = rm_duplicates(text)
            removed_dupl = True
        if write_out_txt: # to change to the processed txt 
            pdf = pdftotext.PDF(pdf, f"OUT_{output_txt_filename}.txt")
        
        try: 
            full_text = " ".join(
                    [
                        od_tup for od_tup in collections.OrderedDict(
                        sorted(text.items())).values()
                    ])
        except Exception as e :
            return None 
        return {
            "file_ext": file_ext, 
            "file_type": file_type,
            "file_content_class": file_content_class,
            "contains_dupl": contains_dupl,
            "removed_dupl": removed_dupl ,
            "doc_text": full_text,
        }
    else:
        print("The img doc is not supportded in v1, please try a pdf file")
        return None 


#load the pdf 
def convert_pdf_to_img(pdf_file):
    return convert_from_path(pdf_file)


def convert_image_to_text(file):
    text = image_to_string(file)
    return text


def get_text_from_any_pdf(pdf_file):
    images = convert_pdf_to_img(pdf_file)
    final_output = {}
    for pg, img in enumerate(images):
        final_output[pg]=convert_image_to_text(img)
    return final_output


def prepare_pdf(
    pdf_path
    ):
    from pdf2image import convert_from_path

    utf8_text = get_text_from_any_pdf(pdf_path)#get text from pdf
    if not os.path.isdir(pdf_path.split('.')[-2]):
        os.mkdir(pdf_path.split('.')[-2])
    else:
        shutil.rmtree(pdf_path.split('.')[-2])
        os.mkdir(pdf_path.split('.')[-2])
    pages = convert_from_path(pdf_path)
    for i in range(len(pages)):
        pages[i].save(
            f"{pdf_path.split('.')[-2]}/page"+ str(i) +'.png', 'png'
        )
    return pdf_path.split('.')[-2]