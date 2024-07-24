
import gradio as gr
from fastai.vision.all import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

pathlib.PosixPath = temp

def what_bear(x): return x[0].isupper()

learn = load_learner('export.pkl')

categories = ('Black', 'Grizzly','Teddy')

def classify_image(img):
    pred, pred_idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=192, width = 192)
label = gr.Label()
title = "Bear Classifier"
description = "A bear classifier trained on images from Bing search using fastai. Able to differentiate from grizzly, black, and teddy bears. Created as a demo for Gradio and HuggingFace Spaces."
examples = ['black.jpg', 'grizzly.jpg','teddy.jpg']

demo = gr.Interface(fn=classify_image, inputs=image, outputs=label, title=title, description=description, examples=examples)
demo.launch()