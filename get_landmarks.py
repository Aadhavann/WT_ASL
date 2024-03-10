import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualize import draw_landmarks_on_image
import cv2

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file("images/img.jpg")

detection_result = detector.detect(image)

for hand_landmarks in detection_result.hand_landmarks:
    print('Hand landmarks:')
    for landmark in hand_landmarks:
        print(landmark)

output_size = (640, 480)

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
annotated_image = cv2.resize(annotated_image, output_size)

cv2.imshow("Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
