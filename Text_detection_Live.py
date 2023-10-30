# Live text detection
# python object_detection_and_speech/text_detection_Live.py --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
from pytesseract import Output
import pytesseract
import pyttsx3


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-d", "--min-conf", type=int, default=0,
	help="mininum confidence value to filter weak text detection")
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# start the FPS throughput estimator
fps = FPS().start()

# initialize pyttsx3
engine =pyttsx3.init()

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

        # check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))

	# construct a blob from the frame and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	
##	# load the input image, convert it from BGR to RGB channel ordering,
##        # and use Tesseract to localize each area of text in the input image
##	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##	results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
##
##	# loop over each of the individual text localizations
##	for i in range(0, len(results["text"])):
##                # extract the bounding box coordinates of the text region from
##                # the current result
##                x = results["left"][i]
##                y = results["top"][i]
##                w = results["width"][i]
##                h = results["height"][i]
##                # extract the OCR text itself along with the confidence of the
##                # text localization
##                text = results["text"][i]
##                conf = int(results["conf"][i])
##                # filter out weak confidence text localizations
##                if conf > args["min_conf"]:
##                        # display the confidence and text to our terminal
##                        print("Confidence: {}".format(conf))
##                        print("Text: {}".format(text))
##                        print("")
##                        # strip out non-ASCII text so we can draw the text on the image
##                        # using OpenCV, then draw a bounding box around the text along
##                        # with the text itself
##                        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
##                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
##                                1.2, (0, 0, 255), 3)
	results = []

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# extract the ROI from the image and pass it through Tesseract
		roi = orig[startY:endY, startX:endX]

		# apply thresholding to remove noise
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

		# apply some blurring and dilation to further remove noise
		blur = cv2.GaussianBlur(thresh, (3,3), 0)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		dilate = cv2.dilate(blur, kernel, iterations=1)
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(dilate, config=config)
		results.append(((startX, startY, endX, endY), text))
                
		# draw the bounding box on the frame
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

	for ((startX, startY, endX, endY), text) in results:
                # add the OCR'd text to the image
                print(text)
                engine.say(text)
                engine.runAndWait()
        # update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
