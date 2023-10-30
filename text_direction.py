import cv2
import pytesseract
import numpy as np

# ...

def detect_text(image):
    # Load the EAST text detection model
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # Preprocess the image for text detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)

    # Forward pass through the network to obtain text detection output
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # Decode the predictions
    boxes, confidences = decode_predictions(scores, geometry)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    # Existing code...

    indices = [int(index) for index in indices]  # Convert indices to integers

    # Access the boxes using the integer indices
    directions = get_directions(image, [boxes[index] for index in indices])


def decode_predictions(scores, geometry):
    # Extract the number of rows and columns from scores
    num_rows, num_cols = scores.shape[2:4]
    num_angles = geometry.shape[2] // 5

    # Initialize the bounding box and confidence lists
    boxes = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]

            if score < 0.5:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0

            angle_index = np.argmax(angles_data)
            angle = angles_data[angle_index]

            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append([start_x, start_y, end_x, end_y])
            confidences.append(score)

    return np.array(boxes), np.array(confidences)

def get_directions(image, boxes):
    # Calculate the midpoint of the image
    midpoint_x = image.shape[1] // 2
    midpoint_y = image.shape[0] // 2

    directions = []
    for box in boxes:
        # Extract the coordinates of the bounding box
        (x, y, w, h, angle) = box

        # Calculate the midpoint of the bounding box
        box_midpoint_x = x + (w // 2)
        box_midpoint_y = y + (h // 2)

        # Determine the direction based on the midpoint positions
        if box_midpoint_x < midpoint_x and box_midpoint_y < midpoint_y:
            directions.append("Top Left")
        elif box_midpoint_x > midpoint_x and box_midpoint_y < midpoint_y:
            directions.append("Top Right")
        elif box_midpoint_x < midpoint_x and box_midpoint_y > midpoint_y:
            directions.append("Bottom Left")
        elif box_midpoint_x > midpoint_x and box_midpoint_y > midpoint_y:
            directions.append("Bottom Right")
        elif box_midpoint_x < midpoint_x:
            directions.append("Left")
        elif box_midpoint_x > midpoint_x:
            directions.append("Right")
        elif box_midpoint_y < midpoint_y:
            directions.append("Top")
        else:
            directions.append("Bottom")

    return directions

def capture_image():
    # Code for capturing the image
    print("Capture the image")

if __name__ == "__main__":
    url = "http://192.168.137.164:4747/video"
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        if not success:
            break

        detect_text(img)
        cv2.imshow("Output", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("c"):
            capture_image()

    cv2.destroyAllWindows()
