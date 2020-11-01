from flask import Flask, flash, redirect, render_template, request, session, abort
from flask import Flask
import pandas as pd
import cv2
import glob

import imutils
import numpy as np
import os
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


from collections import Counter
import warnings
from tqdm import tqdm
nltk.download('punkt')

import speech_recognition as sr
import csv


app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = APP_ROOT+'\\media'
for filename in os.listdir(UPLOAD_FOLD):
    file_path = os.path.join(UPLOAD_FOLD, filename)
    os.remove(file_path)
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("file_upload.html")

@app.route('/success', methods = ['GET', 'POST'])
def success():
    if request.method == 'POST':
        for f in request.files.getlist('datafile[]'):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

        DIRNAME = APP_ROOT+'\\media'
            
        OUTPUTFILE = APP_ROOT+'\\media\\outputfile.csv'

        def get_file_paths(dirname):
            file_paths = []
            for root, directories, files in os.walk(dirname):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)
            return file_paths

        def process_file(file):
            r = sr.Recognizer()
            a = ''
            with sr.AudioFile(file) as source:
                audio = r.record(source)
                try:
                    a =  r.recognize_google(audio)
                except sr.UnknownValueError:
                    a = "Speech Recognition could not understand audio"
                except sr.RequestError as e:
                    a = "Could not request results from Speech Recognition service; {0}".format(e)
            return a


        greetings = []
        filenamee = []
        greeting_word = []
        files = get_file_paths(DIRNAME)                 # get all file-paths of all files in dirname and subdirectories
        for file in files:                              # execute for each file
            (filepath, ext) = os.path.splitext(file)    # get the file extension
            file_name = os.path.basename(file)          # get the basename for writing to output file
            filenamee.append(file_name)
            if ext == '.wav':                           # only interested if extension is '.wav'
                a = process_file(file)                  # result is returned to a
                greetings.append(a)
            
            print(a)
            if "good morning" in a: 
                print('Wow.. greeting is there..')
                greeting_word.append('Wow.. greeting is there..')
            else:
                print('Sorry.. No greetings...')
                greeting_word.append('Sorry.. No greetings...')
        
        dff = pd.DataFrame({"Audio":filenamee, "text":greetings, "Result":greeting_word})
        dff.index= dff.index+1
        #from pandas import ExcelWriter

        #writer = ExcelWriter(APP_ROOT+'\\media\\final.xlsx')
        #dff.to_excel(writer,'Sheet5')
        #writer.save()


        return render_template("success.html", tables=[dff.to_html(classes='data')], titles=dff.columns.values)

        
        #dft = sd1.style.applymap(lambda x: 'color: red' if x == "False" else 'color: black')
        #return render_template("success.html", tables=[dft.render(classes='data')], titles=dft.columns.values)

if __name__ == "__main__":
    app.run()
