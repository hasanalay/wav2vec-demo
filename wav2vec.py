# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:20:23 2022

@author: Hasan
"""
# Import necessary library

# For managing audio file
import librosa

#Importing Pytorch
import torch

#Importing Wav2Vec tokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Reading taken audio clip

import IPython.display as display
display.Audio("taken_clip.wav", autoplay=True)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Loading the audio file

audio, rate = librosa.load("taken_clip.wav", sr = 16000)


# printing audio 
print(audio)
print(type(audio))

print(rate)


# Taking an input value 
input_values = tokenizer(audio, return_tensors = "pt").input_values

# Storing logits (non-normalized prediction values)
logits = model(input_values).logits


# Storing predicted ids
prediction = torch.argmax(logits, dim = -1)


# Passing the prediction to the tokenzer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]


# Printing the transcription
print(transcription) 
# 'I WILL LOOK FOR YOU I WILL FIND YOU AND I WILL KILL YOU'

