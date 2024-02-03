# Assignment_3
Extraction of Business Card
 prompt: Extracting business card 

# Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('Desktop/Guvi/Python_1/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image)

# Apply thresholding to binarize the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours
contours = [c for c in contours if cv2.contourArea(c) > 100]

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Crop the image to the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)
business_card = image[y:y + h, x:x + w]

# Display the extracted business card
plt.imshow(business_card, cmap='gray')
plt.show()
