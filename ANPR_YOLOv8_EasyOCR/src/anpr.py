# ANPR System - Automatic Number Plate Recognition
# Author: Rohit Kumar
# Tech: YOLOv8 + EasyOCR + OpenCV

from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import re
import os
import csv
from datetime import datetime


class ANPRConfig:
    YOLO_MODEL = "yolov8n.pt"
    VEHICLE_CONF = 0.5
    OCR_LANGUAGES = ["en"]
    VEHICLE_IDS = [2, 3, 5, 7]
    VEHICLE_CLASSES = {2:"car",3:"motorcycle",5:"bus",7:"truck"}
    CHAR_TO_INT = {"O":"0","B":"8","I":"1","S":"5","Z":"2"}
    INT_TO_CHAR = {"0":"O","8":"B","1":"I","5":"S","2":"Z"}
    OUTPUT_DIR = "output"


class ImagePreprocessor:
    def preprocess(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (3,3), 0)
        equalized = cv2.equalizeHist(blurred)
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binary


class PlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.config = ANPRConfig()

    def correct_plate(self, text):
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        corrected = list(text)
        for i, char in enumerate(corrected):
            if i in [0,1,4,5]:
                if char in self.config.INT_TO_CHAR:
                    corrected[i] = self.config.INT_TO_CHAR[char]
            elif i in [2,3,6,7,8,9]:
                if char in self.config.CHAR_TO_INT:
                    corrected[i] = self.config.CHAR_TO_INT[char]
        return "".join(corrected)

    def validate(self, text):
        pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"
        return bool(re.match(pattern, text))

    def read(self, plate_img):
        results = self.reader.readtext(
            plate_img,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        raw = "".join([r[1] for r in results]).upper()
        corrected = self.correct_plate(raw)
        return {"raw": raw, "text": corrected, "valid": self.validate(corrected)}


class ANPRSystem:
    def __init__(self):
        self.config = ANPRConfig()
        self.model = YOLO(self.config.YOLO_MODEL)
        self.preprocessor = ImagePreprocessor()
        self.plate_reader = PlateReader()
        self.plate_count = 0
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def process_video(self, video_path, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        while cap.isOpened() and frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            results = self.model(
                frame,
                conf=self.config.VEHICLE_CONF,
                classes=self.config.VEHICLE_IDS,
                verbose=False
            )
            print(f"Frame {frame_num}: {len(results[0].boxes)} vehicles detected")
        cap.release()
        print(f"Processing complete. Frames: {frame_num}")


if __name__ == "__main__":
    print("ANPR System Starting...")
    anpr = ANPRSystem()
    print("System Ready!")
    print("Pipeline: Video -> YOLO -> Plate Detection -> OCR -> Post Processing")
