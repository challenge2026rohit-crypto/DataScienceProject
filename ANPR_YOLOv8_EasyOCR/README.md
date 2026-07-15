# Automatic Number Plate Recognition ANPR
## YOLOv8 + EasyOCR

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0-orange)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7-green)

## Project Overview
End-to-end ANPR system using deep learning.
Detects vehicles and reads number plates from video in real time.

## Pipeline
Video Input
  -> Vehicle Detection using YOLOv8
  -> Number Plate Region Detection using Contour Analysis
  -> Image Preprocessing using OpenCV
     Grayscale -> Resize 2x -> Denoise -> OTSU Threshold
  -> Text Reading using EasyOCR
  -> Post Processing using Indian Plate Format Rules
  -> Result: Plate number + Validity check

## Performance Results

| Metric | Score |
|--------|-------|
| Vehicle Detection | 94% |
| Raw OCR Accuracy | 70% |
| After Post Processing | 87% |
| Improvement | +17% |
| Processing Speed | 30 FPS |

## Key Innovation
Implemented Indian plate format rules for character correction:
- Positions 0,1,4,5 must be letters: 0 becomes O, 8 becomes B
- Positions 2,3,6-9 must be numbers: O becomes 0, B becomes 8
- This improved OCR accuracy from 70% to 87%

## Tech Stack
- YOLOv8 for vehicle detection
- EasyOCR for text recognition
- OpenCV for image preprocessing
- Python for post processing logic

## Project Structure

ANPR_YOLOv8_EasyOCR
  src
    anpr.py          <- Main ANPR pipeline
  results
    detected_plates  <- Saved plate images
  docs
    pipeline.md      <- System documentation
  sample_data        <- Test videos
  requirements.txt
  README.md

## How to Run

Install:
  pip install -r requirements.txt

Run:
  python src/anpr.py

## Author
Rohit Kumar
- IIT Kanpur E&ICT Certified Data Scientist
- 500+ students trained
- Email: rk2401ds@gmail.com
- LinkedIn: linkedin.com/in/kumarrohit24