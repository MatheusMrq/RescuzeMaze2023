import cv2
import numpy as np

def detect_color(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for green in HSV format
    green_low = np.array([35, 100, 100])
    green_high = np.array([85, 255, 255])

    # Define color ranges for red in HSV format
    red_low = np.array([0, 100, 100])
    red_high = np.array([10, 255, 255])

    # Define color ranges for yellow in HSV format
    yellow_low = np.array([20, 100, 100])
    yellow_high = np.array([30, 255, 255])

    # Create masks for each of the colors
    green_mask = cv2.inRange(hsv_frame, green_low, green_high)
    red_mask = cv2.inRange(hsv_frame, red_low, red_high)
    yellow_mask = cv2.inRange(hsv_frame, yellow_low, yellow_high)

    # Find contours in the masks
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any of the colors were detected
    if len(green_contours) > 0:
        return "G"
    elif len(red_contours) > 0:
        return "R"
    elif len(yellow_contours) > 0:
        return "Y"
    else:
        return ""

def detect_letter(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to segment the black objects
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Invert the colors so that objects are white and the background is black
    inverted_thresholded = cv2.bitwise_not(thresholded)

    # Find contours in the binary image
    contours, _ = cv2.findContours(inverted_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the calculated Hu moments
    hu_moments_list = []

    for contour in contours:
        # Calculate Hu moments from the contour
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Add the Hu moments to the list
        hu_moments_list.append(hu_moments)

    if hu_moments_list:
        # Calculate the average Hu moments from the list
        avg_hu_moments = np.mean(hu_moments_list, axis=0)

        # Calculate the difference between the average Hu moments and a reference object
        difference = np.abs(avg_hu_moments - reference_hu_moments).sum()

        # Round the difference to 2 decimal places
        rounded_difference = round(difference, 2)

        # Check if the current value matches the previous value
        if 3.55 <= rounded_difference < 3.66:
            return "S"
        elif 3.67 <= rounded_difference <= 3.74:
            return "H"
        elif 3.75 <= rounded_difference <= 3.81:
            return "U"

    return ""

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Hu moments of a reference object (adjust as needed)
reference_hu_moments = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

while True:
    # Capture a video frame
    ret, frame = cap.read()

    # Display the frame with contours
    cv2.imshow('Color and Letter Detection', frame)

    # Detect the pressed key
    key = cv2.waitKey(1) & 0xFF

    color_result = detect_color(frame)
    letter_result = detect_letter(frame)

    if key == ord('L') or key == ord('l'):
        # Perform reading of information
        if color_result:
            print("Color Detected:", color_result)
        if letter_result:
            print("Letter Detected:", letter_result)
        if not color_result and not letter_result:
            print("X")  # Send "X" if no color or letter is detected

    elif key == ord('R') or key == ord('r'):
        # Detect the color red
        if color_result == "R":
            print("Color Detected: Red")
        else:
            print("No red color detected")

    elif key == ord('G') or key == ord('g'):
        # Detect the color green
        if color_result == "G":
            print("Color Detected: Green")
        else:
            print("No green color detected")

    elif key == ord('Y') or key == ord('y'):
        # Detect the color yellow
        if color_result == "Y":
            print("Color Detected: Yellow")
        else:
            print("No yellow color detected")

    elif key == ord('S') or key == ord('s'):
        # Detect the letter 'S'
        if letter_result == "S":
            print("Letter Detected: S")
        else:
            print("No letter 'S' detected")

    elif key == ord('U') or key == ord('u'):
        # Detect the letter 'U'
        if letter_result == "U":
            print("Letter Detected: U")
        else:
            print("No letter 'U' detected")

    elif key == ord('X') or key == ord('x'):
        print("No action executed")

    elif key == ord('Q') or key == ord('q'):
        break

# Release video capture and close the window
cap.release()
cv2.destroyAllWindows()
