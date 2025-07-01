import cv2 as cv
import numpy as np

# function that takes url as parameter and returns the final processed frame
def frame_capture(url):

    #using ip webcam,this is used to get access to the stream
    cap = cv.VideoCapture(url, cv.CAP_FFMPEG)

    if cap.isOpened():
        print("connected to stream")
        while True:
        
            print("press c to capture the frame or press q to quit")

            #return is a boolean which returns True if the frame was read and false otherwise while frame is the actual frame that was captured
            ret, frame = cap.read()
            if not ret:
                print(" Failed to grab frame")

            cv.imshow("Phone Camera Feed", frame)
            key = cv.waitKey(1)
            print("waiting for key to be pressed ")

            #will capture frame when 'c' key is pressed and then preprocess that image
            if key == ord("c"):  
                filename = "frame.jpg"
                cv.imwrite(filename, frame)
                print("test_input captured")
                gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
                _, thresh = cv.threshold(gray, 75, 255, cv.THRESH_BINARY_INV)
                contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                cnt = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(cnt)
                digit = thresh[y:y+h, x:x+w]
                max_dim = max(w, h)
                square_digit = np.zeros((max_dim, max_dim), dtype=np.uint8)
                x_offset = (max_dim - w) // 2
                y_offset = (max_dim - h) // 2
                square_digit[y_offset:y_offset+h, x_offset:x_offset+w] = digit
                final_img = cv.resize(square_digit, (28, 28), interpolation=cv.INTER_AREA)
                enlarged = cv.resize(final_img, (280, 280), interpolation=cv.INTER_NEAREST)
                cv.imshow("captured_frame",enlarged)
                cv.waitKey(0)
                test_input = final_img.astype(np.float32) / 255.0
                test_input=np.reshape(test_input,(784,1))
                break
            
        #press q to quit
            elif key == ord("q"):  
                print("Exiting...")
                break

        cap.release()
        cv.destroyAllWindows()

        # in case no frame was captured,it shouldn't return anything
        try:
            return test_input
        except NameError:
            print("No frame was captured.")
            return None


    else:
        print("could not connect to webcam stream")
        return None
            

    