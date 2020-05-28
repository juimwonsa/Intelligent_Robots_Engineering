# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:34:26 2020

@author: gmlgn
"""

import speech_recognition as sr

r = sr.Recognizer()

harvard = sr.AudioFile('harvard.wav')
with harvard as source:
    audio = r.record(source)

type(audio)

print(r.recognize_google(audio))
