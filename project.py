import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("models/yolov8m-seg")

img_path = "res/test_1.jpg"
results = model.predict(img_path)

img = cv2.imread(img_path)
output = np.zeros_like(img)
print(output.shape)

mask = results[0].masks.masks[0].numpy()

mask_inv = 1 - mask
mask_inv = cv2.resize(mask_inv, (output.shape[1], output.shape[0]))

output[mask_inv == 1] = [255, 255, 255]

output = cv2.bitwise_or(output, img)

cv2.imwrite("res/test_1_out.jpg", output)
cv2.waitKey(0)
cv2.destroyAllWindows()