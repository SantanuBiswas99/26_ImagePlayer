from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import image as img
import cv2
import numpy as np
import shutil
from scipy.interpolate import UnivariateSpline
from collections import deque
import cv2 as cv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import sys
import logging as log
import datetime as dt
from time import sleep
import os
import subprocess
import requests
import webbrowser
from googlesearch import search   


def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))
def alpha_blend(image_1, image_2, mask):
    alpha = mask/255.0 
    blended = cv2.convertScaleAbs(image_1*(1-alpha) + image_2*alpha)
    return blended
def verify_alpha_channel(image):
    try:
        image.shape[3] 
    except IndexError:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image

def invert_util(image):
	return cv2.bitwise_not(image)
def cartoon_util(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
def graySketch_util(image):
    gray_sketch, color_sketch=cv2.pencilSketch(image, sigma_s=20, sigma_r=0.25 , shade_factor=0.02)
    return gray_sketch
def colourSketch_util(image):
    gray_sketch, color_sketch=cv2.pencilSketch(image, sigma_s=20, sigma_r=0.25 , shade_factor=0.02)
    return color_sketch
def detailEnhance_util(image):
    image=cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)
    return image
def changeBackground_util(image):
    segmentor = SelfiSegmentation()
    bg= (255, 0, 0)
    image=segmentor.removeBG(image, bg, threshold=0.50)
    return image
def alpha_blend_util(image_1, image_2, mask):
    alpha = mask/255.0 
    blended = cv2.convertScaleAbs(image_1*(1-alpha) + image_2*alpha)
    return blended
def verify_alpha_channel_util(image):
    try:
        image.shape[3] 
    except IndexError:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image
def circular_blur_util(image, intensity=0.2):
    image = verify_alpha_channel(image)
    image_h, image_w, image_c = image.shape
    y = int(image_h/2)
    x = int(image_w/2)
    mask = np.zeros((image_h, image_w, 4), dtype='uint8')
    cv2.circle(mask, (x, y), int(y/2), (255,255,255), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21,21),11 )
    blured = cv2.GaussianBlur(image, (21,21), 11)
    blended = alpha_blend(image, blured, 255-mask)
    image = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return image


app = Flask(__name__)
 
@app.route('/')
def hello():
    return render_template('first.html')

@app.route('/cards')
def cards():
    return render_template('cards.html')

@app.route('/basic')
def basic():
    return render_template('main.html')

@app.route('/status')
def status():
    return render_template('status.html')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/status2', methods = ['GET', 'POST'])
def status2():
    if request.method == 'POST':
        f = request.files['file']
        f.filename = 'uploaded_image.jpg'
        f.save(secure_filename(f.filename))
        shutil.copy('uploaded_image.jpg','static/')
        return render_template('status2.html')
    
@app.route('/bilateral')
def bilateral():
    image = cv2.imread('uploaded_image.jpg')
    image = np.array(image,dtype='uint8')
    #Image processing
    blur = cv2.bilateralFilter(image,9,75,75)  
    cv2.imwrite('processed_image.jpg',blur)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')



@app.route('/cartoon')
def cartoon():
    image = cv2.imread(r'uploaded_image.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite('processed_image.jpg',cartoon)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/cartoon_video')
def cartoon_video():
    cap = cv.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        frame = cartoon_util(image)
        cv.imshow("Video", frame)
        k = cv.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

@app.route('/inverted')
def inverted():
    image = cv2.imread(r'uploaded_image.jpg')
    image = cv2.bitwise_not(image)
    cv2.imwrite('processed_image.jpg',image)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/invert_video')
def invert_video():
    cap = cv.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        frame = invert_util(image)
        cv.imshow("Video", frame)
        k = cv.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

@app.route('/changeBG_video')
def changeBG_video():
    cap = cv.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        frame = changeBackground_util(image)
        cv.imshow("Video", frame)
        k = cv.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

@app.route('/gray_sketch')
def graySketch():
    image = cv2.imread(r'uploaded_image.jpg')
    gray_sketch, color_sketch=cv2.pencilSketch(image, sigma_s=20, sigma_r=0.25 , shade_factor=0.02)
    cv2.imwrite('processed_image.jpg',gray_sketch)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/graySketch_video')
def graySketch_video():
    cap = cv.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        frame = graySketch_util(image)
        cv.imshow("Video", frame)
        k = cv.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

@app.route('/colourSketch_video')
def colourSketch_video():
    cap = cv.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        frame = colourSketch_util(image)
        cv.imshow("Video", frame)
        k = cv.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

@app.route('/color_sketch')
def colorSketch():
    image = cv2.imread(r'uploaded_image.jpg')
    gray_sketch, color_sketch=cv2.pencilSketch(image, sigma_s=20, sigma_r=0.25 , shade_factor=0.02)
    cv2.imwrite('processed_image.jpg',color_sketch)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/detail_enhance')
def detailEnhance():
    image = cv2.imread(r'uploaded_image.jpg')
    detail=cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)
    cv2.imwrite('processed_image.jpg',detail)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/detailEnhance_video')
def detailEnhance_video():
    cap = cv.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        frame = detailEnhance_util(image)
        cv.imshow("Video", frame)
        k = cv.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

@app.route('/summer')
def summer():
    image = cv2.imread(r'uploaded_image.jpg')
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    image= cv2.merge((blue_channel, green_channel, red_channel ))
    cv2.imwrite('processed_image.jpg',image)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/winter')
def winter():
    image = cv2.imread(r'uploaded_image.jpg')
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    image= cv2.merge((blue_channel, green_channel, red_channel))
    cv2.imwrite('processed_image.jpg',image)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')



@app.route('/remove_background')
def removeBackground():
    image = cv2.imread(r'uploaded_image.jpg')
    segmentor = SelfiSegmentation()
    bg= (255, 0, 0)
    image=segmentor.removeBG(image, bg, threshold=0.50)
    cv2.imwrite('processed_image.jpg',image)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/circular_blur')
def circular_blur():
    intensity=0.2
    image = cv2.imread(r'uploaded_image.jpg')
    image = verify_alpha_channel(image)
    image_h, image_w, image_c = image.shape
    y = int(image_h/2)
    x = int(image_w/2)
    mask = np.zeros((image_h, image_w, 4), dtype='uint8')
    cv2.circle(mask, (x, y), int(y/2), (255,255,255), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21,21),11 )
    blured = cv2.GaussianBlur(image, (21,21), 11)
    blended = alpha_blend(image, blured, 255-mask)
    image = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    cv2.imwrite('processed_image.jpg',image)
    shutil.copy('processed_image.jpg','static/')
    return render_template('status3.html')

@app.route('/draw')
def draw():
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    blackboard_copy = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    pred_class = 0
    cap = cv.VideoCapture(0)
    while (cap.isOpened()):
            ret, image = cap.read()
            image = cv.flip(image, 1)
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            kernel = np.ones((5, 5), np.uint8)
            Lower_green = np.array([110 ,50, 50])
            Upper_green = np.array([130, 255, 255])

            mask = cv.inRange(hsv, Lower_green, Upper_green)
            mask = cv.erode(mask, kernel, iterations=2)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
            mask = cv.dilate(mask, kernel, iterations=1)
            res = cv.bitwise_and(image, image, mask=mask)
            cnts, heir = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
            center = None

            if len(cnts) >= 1:
                cnt = max(cnts, key=cv.contourArea)
                if cv.contourArea(cnt) > 200:
                    ((x, y), radius) = cv.minEnclosingCircle(cnt)
                    cv.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(image, center, 5, (0, 0, 255), -1)
                    M = cv.moments(cnt)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                    pts.appendleft(center)
                    for i in range(1, len(pts)):
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                        cv.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                        cv.line(image, pts[i - 1], pts[i], (0, 0, 255), 2)

            cv.imshow("Frame", image)
            
            #blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            #Setting hsv value for red
            lower_red = np.array([0,70,50])
            upper_red = np.array([10, 255, 255])

            #Converting into hsv and then applying masking
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower_red, upper_red)

            cv.imshow("black", mask)
            k = cv.waitKey(10)
            if k == 27:
                break

    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',image)
    cv2.imwrite('blackImg.jpg',mask)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

drawing = False 
mode = False 
ix,iy = -1,-1

@app.route('/mouse')
def mouse():
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,mode
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv.rectangle(blackboard,(ix,iy),(x,y),(0,255,0),-1)
                else:
                    cv.circle(blackboard,(x,y),5,(0,0,255),-1)
        elif event == cv.EVENT_LBUTTONUP:
                drawing = False
                if mode == True:
                    cv.rectangle(blackboard,(ix,iy),(x,y),(0,255,0),-1)
                else:
                    cv.circle(blackboard,(x,y),5,(0,0,255),-1)
                    
    
    
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)
    #Writing on blackboard
    while(1):
        cv.imshow('image',blackboard)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',blackboard)
    shutil.copy('processed_image.jpg','static/')

    api_url = 'https://api.api-ninjas.com/v1/imagetotext'
    image_file_descriptor = open('processed_image.jpg', 'rb')
    files = {'image': image_file_descriptor}
    r = requests.post(api_url, files=files,headers={'X-Api-Key': 'JnEDyuGJJHZthBr5NGAAYw==VhROx0Y25gqQl1eu'})
    res = r.json()
    if(len(res)>0):
        keyword= res[0]['text']
        url = "https://google.com/search?q=" + keyword
        webbrowser.open(url)

        links = []
        for j in search(keyword, tld="co.in", num=10, stop=10, pause=2): 
            links.append(j)
            if len(links) > 1:
                break

        for i in range(len(links)):
            webbrowser.open(links[i])
    return render_template('videos2.html')


@app.route('/swag')
def swag():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    specs_ori = cv2.imread('glass.png', -1)
    cigar_ori = cv2.imread('cigar.png', -1)
    mus_ori = cv2.imread('mustache.png', -1)

    # Camera Init
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)


    def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape  # Size of foreground
        rows, cols, _ = src.shape  # Size of background Image
        y, x = pos[0], pos[1]  # Position of foreground/overlay image

        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
                src[x + i][y + j] = alpha * overlay[i][j][:3] + \
                    (1 - alpha) * src[x + i][y + j]
        return src


    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            img, 1.2, 5, 0, (120, 120), (350, 350))
        for (x, y, w, h) in faces:
            if h > 0 and w > 0:
                glass_symin = int(y + 1.5 * h / 5)
                glass_symax = int(y + 2.5 * h / 5)
                sh_glass = glass_symax - glass_symin

                cigar_symin = int(y + 4 * h / 6)
                cigar_symax = int(y + 5.5 * h / 6)
                sh_cigar = cigar_symax - cigar_symin

                mus_symin = int(y + 3.5 * h / 6)
                mus_symax = int(y + 5 * h / 6)
                sh_mus = mus_symax - mus_symin

                face_glass_roi_color = img[glass_symin:glass_symax, x:x + w]
                face_cigar_roi_color = img[cigar_symin:cigar_symax, x:x + w]
                face_mus_roi_color = img[mus_symin:mus_symax, x:x + w]

                specs = cv2.resize(specs_ori, (w, sh_glass),
                                   interpolation=cv2.INTER_CUBIC)
                cigar = cv2.resize(cigar_ori, (w, sh_cigar),
                                   interpolation=cv2.INTER_CUBIC)
                mustache = cv2.resize(mus_ori, (w, sh_mus),
                                      interpolation=cv2.INTER_CUBIC)

                transparentOverlay(face_glass_roi_color, specs)
                transparentOverlay(face_cigar_roi_color, cigar,(int(w / 2), int(sh_cigar / 2)))
                transparentOverlay(face_mus_roi_color, mustache)

        cv2.imshow('Thug Life', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',img)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')


@app.route('/hat')
def hat():
    cascPath = "haarcascade_frontalface_default.xml"  # for face detection

    faceCascade = cv2.CascadeClassifier(cascPath)
    log.basicConfig(filename='webcam.log', level=log.INFO)

    video_capture = cv2.VideoCapture(0)
    anterior = 0
    hat = cv2.imread('cowboy_hat.png')

    def put_hat(hat, fc, x, y, w, h):
        face_width = w
        face_height = h

        hat_width = face_width+1
        hat_height = int(0.35*face_height)+1

        hat = cv2.resize(hat, (hat_width, hat_height))

        for i in range(hat_height):
            for j in range(hat_width):
                for k in range(3):
                    if hat[i][j][k] < 235:
                        fc[y+i-int(0.25*face_height)][x+j][k] = hat[i][j][k]
        return fc

    ch = 1
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:

            frame = put_hat(hat, frame, x, y, w, h)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',frame)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')


@app.route('/removeBG_video')
def removeBG_video():
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(cv2.CAP_PROP_FPS, 60)

    segmentor = SelfiSegmentation()
    fpsReader = cvzone.FPS()

    # imgBG = cv2.imread("BackgroundImages/3.jpg")

    imgList = []
    img = cv2.imread(f'1.jpg')
    imgList.append(img)
    indexImg = 0
    while True:
        success, img = cap.read()
        # imgOut = segmentor.removeBG(img, (255,0,255), threshold=0.83)
        imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)

        imgStack = cvzone.stackImages([imgOut], 1,1)
        #_, imgStack = fpsReader.update(imgStack)
        cv2.imshow("image", imgStack)
        key = cv2.waitKey(1)
        if key == ord('a'):
            if indexImg>0:
                indexImg -=1
        elif key == ord('d'):
            if indexImg<len(imgList)-1:
                indexImg +=1
        elif key == ord('q'):
            break
        k = cv.waitKey(10)
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    cv2.imwrite('processed_image.jpg',imgStack)
    shutil.copy('processed_image.jpg','static/')
    return render_template('videos2.html')

if __name__ == '__main__':
    app.run(port=5005, threaded=False,debug=False)