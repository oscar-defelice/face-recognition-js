# Face Recognition
This repository contains the code to implement a smart webcam in JavaScript with a pre-trained Machine Learning model fine-tuned to recognise faces.

## [Click me for Live Demos!](https://oscar-defelice.github.io/face-recognition-js)

## Introduction

This repository contains the code useful to publish a web API to take advantage of the webcam to perform object recognition. The model is a pre-trained model, fine-tuned, then converted in and deployed in [tensorflow-js](https://www.tensorflow.org/js/models).

### The model
We make use of the [face-api.js](https://github.com/justadudewhohacks/face-api.js) that in turns uses the SSD Mobilenet V1 Face Detector.

We fine-tune the model by a training routine.

### Usage

Simply open the [webpage](https://oscar-defelice.github.io/face-recognition-js) and smile!
You can optionally insert the name of the person to recognise.
