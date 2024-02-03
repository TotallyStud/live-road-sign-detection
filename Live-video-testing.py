#importing libraries
import numpy as np
import cv2
from tensorflow import keras

#loading the model
threshold = 0.75  # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model = keras.models.load_model('traffif_sign_model.h5')

#function to preprocess the image
def preprocess_img(imgBGR, erode_dilate=True):  # pre-processing fro detect signs in  image.
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin

#Counting the number of signs in the image
def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

#preprocessing the image before feeding it to the model
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

#Defining the labels
def getCalssName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'
    

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce resolution
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    while True:
        ret, img = cap.read()
        frame_count += 1
        if frame_count % 2 == 0:  # Skip every other frame
            continue
        img_bin = preprocess_img(img, False)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)
        img_bbx = img.copy()
        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)
            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            if rect[2] > 100 and rect[3] > 100:
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
            crop_img = np.asarray(img[y1:y2, x1:x2])
            crop_img = cv2.resize(crop_img, (32, 32))
            crop_img = preprocessing(crop_img)
            crop_img = crop_img.reshape(1, 32, 32, 1)
            predictions = model.predict(crop_img)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.amax(predictions)
            if probabilityValue > threshold:
                cv2.putText(img_bbx, str(classIndex) + " " + str(getCalssName(classIndex)), (rect[0], rect[1] - 10),
                            font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                            (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Output", img_bbx)  # Display the output

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Stop the program by pressing ''
            break
    cap.release()
    cv2.destroyAllWindows()

    







    # The code below is the same as the code above, but it is commented out because it is not necessary for the final project. 
    #it includes for grey scale, equalize, and data augmentation.
    
    
    """
    if __name__ == "__main__":
    # Initialize video capture with webcam
    cap = cv2.VideoCapture(0)
    # Get the video frame's width and height
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        # Read frame from the video
        ret, img = cap.read()
        # Preprocess the image
        img_bin = preprocess_img(img, False)
        # Display the preprocessed image
        cv2.imshow("bin image", img_bin)
        # Define minimum area for detected sign
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        # Detect contours in the image
        rects = contour_detect(img_bin, min_area=min_area)
        # Copy the original image
        img_bbx = img.copy()
        for rect in rects:
            # Calculate center of the rectangle
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            # Get the size of the rectangle
            size = max(rect[2], rect[3])
            # Calculate coordinates for cropping
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            # Only detect signs with width and height greater than 100
            if rect[2] > 100 and rect[3] > 100:
                # Draw rectangle around the detected sign
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
            # Crop the image
            crop_img = np.asarray(img[y1:y2, x1:x2])
            # Resize the cropped image
            crop_img = cv2.resize(crop_img, (32, 32))
            # Preprocess the cropped image
            crop_img = preprocessing(crop_img)
            # Display the preprocessed cropped image
            cv2.imshow("afterprocessing", crop_img)
            # Reshape the image to match the model's input shape
            crop_img = crop_img.reshape(1, 32, 32, 1)
            # Make prediction with the model
            predictions = model.predict(crop_img)
            # Get the index of the class with the highest probability
            classIndex = np.argmax(predictions, axis=-1)
            # Get the highest probability
            probabilityValue = np.amax(predictions)
            # If the probability is higher than the threshold
            if probabilityValue > threshold:
                # Write the class name on the image
                cv2.putText(img_bbx, str(classIndex) + " " + str(getCalssName(classIndex)), (rect[0], rect[1] - 10),
                            font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                # Write the probability on the image
                cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                            (0, 0, 255), 2, cv2.LINE_AA)

        # Display the image with detected signs
        cv2.imshow("detect result", img_bbx)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    """