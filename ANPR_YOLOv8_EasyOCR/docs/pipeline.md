# ANPR Pipeline Documentation

## Step 1: Vehicle Detection
YOLOv8 detects vehicles in each video frame.
Returns bounding boxes for cars, bikes, buses, trucks.

## Step 2: Plate Detection
Contour analysis finds rectangular regions.
Filters by aspect ratio 1.5 to 6.0 (plate shape).

## Step 3: Preprocessing
1. Grayscale conversion
2. 2x upscaling for better OCR
3. Gaussian blur for noise removal
4. Histogram equalization for contrast
5. OTSU thresholding for binarization

## Step 4: OCR
EasyOCR reads text from preprocessed plate image.
Restricted to alphanumeric characters only.

## Step 5: Post Processing
Indian plate format: SS DD LL NNNN
Positions 0,1,4,5 must be letters.
Positions 2,3,6-9 must be numbers.
Character corrections applied by position.
Improves accuracy from 70% to 87%.