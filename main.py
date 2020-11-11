from string import ascii_uppercase,digits
import pytesseract as pyt
import cv2
import numpy as np
from PIL import ImageGrab
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def objectDetector(img):
    yolo = cv2.dnn.readNet("model.weights", "darknet-yolov3.cfg") # my yolo's file path
    classes = []
    with open("classes.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
    height, width, channels = img.shape
    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            sem_placa = False
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            else:
                sem_placa = True
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            y = y-(int(y*0.05))
            w = w+(int(w*0.24))
            h = h+(int(h*0.73))
            x = x-(int(x*0.02))

            #label = str(classes[class_ids[i]])
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 3)
            #cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 3, colorWhite, 2)
    return img, x, y, w, h,sem_placa
def nothing(x):
    pass
def captura_coordenadas():
    x_begin = cv2.getTrackbarPos("x_begin", "Trackbars")
    y_begin = cv2.getTrackbarPos("y_begin", "Trackbars")
    x_end = cv2.getTrackbarPos("x_end", "Trackbars")
    y_end = cv2.getTrackbarPos("y_end", "Trackbars")

    x_b_captura = cv2.getTrackbarPos("x_b_captura", "Trackbars1")
    y_b_captura = cv2.getTrackbarPos("y_b_captura", "Trackbars1")
    x_e_captura = cv2.getTrackbarPos("x_e_captura", "Trackbars1")
    y_e_captura = cv2.getTrackbarPos("y_e_captura", "Trackbars1")
    return x_begin,y_begin,x_end,y_end,x_b_captura,y_b_captura,x_e_captura,y_e_captura
def cria_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.namedWindow("Trackbars1")
    #cv2.moveWindow("Trackbars", 70, 0)
    #cv2.moveWindow("Trackbars1", 320, 0)
    cv2.createTrackbar("x_begin", "Trackbars", 76, 1366 + 1440, nothing)
    cv2.createTrackbar("y_begin", "Trackbars", 155, 768 + 900, nothing)
    cv2.createTrackbar("x_end", "Trackbars", 1295, 1366 + 1440, nothing)
    cv2.createTrackbar("y_end", "Trackbars", 671, 768 + 900, nothing)
    cv2.createTrackbar("x_b_captura", "Trackbars1", 735, 1366 + 1440, nothing)
    cv2.createTrackbar("y_b_captura", "Trackbars1", 253, 768 + 900, nothing)
    cv2.createTrackbar("x_e_captura", "Trackbars1", 135, 1366 + 1440, nothing)
    cv2.createTrackbar("y_e_captura", "Trackbars1", 276, 768 + 900, nothing)

    cv2.namedWindow("TrackbarsHSV")
    cv2.moveWindow("TrackbarsHSV", 1367, 0)
    '''
    #original
    cv2.createTrackbar("L-H", "TrackbarsHSV", 0, 180, nothing)
    cv2.createTrackbar("L-S", "TrackbarsHSV", 0, 255, nothing)
    cv2.createTrackbar("L-V", "TrackbarsHSV", 47, 255, nothing)
    cv2.createTrackbar("U-H", "TrackbarsHSV", 0, 180, nothing)
    cv2.createTrackbar("U-S", "TrackbarsHSV", 51, 255, nothing)
    cv2.createTrackbar("U-V", "TrackbarsHSV", 255, 255, nothing)
    '''
    cv2.createTrackbar("L-H", "TrackbarsHSV", 0, 180, nothing)
    cv2.createTrackbar("L-S", "TrackbarsHSV", 0, 255, nothing)
    cv2.createTrackbar("L-V", "TrackbarsHSV", 98, 255, nothing)
    cv2.createTrackbar("U-H", "TrackbarsHSV", 180, 180, nothing)
    cv2.createTrackbar("U-S", "TrackbarsHSV", 255, 255, nothing)
    cv2.createTrackbar("U-V", "TrackbarsHSV", 130, 255, nothing)



def captura_tela(x_begin,y_begin,x_end,y_end,x_b_captura,y_b_captura,x_e_captura,y_e_captura):
    frame = np.array(ImageGrab.grab(bbox=(x_begin, y_begin, x_end, y_end)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cv2.rectangle(frame,(x_b_captura,y_b_captura),(x_b_captura+x_e_captura,y_b_captura+y_e_captura),(0,0,255),3)
    return frame
if __name__ == '__main__':
    pyt.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' #my tesseracts defaulf path
    font = cv2.FONT_HERSHEY_COMPLEX
    fgbg = cv2.createBackgroundSubtractorMOG2(15, 400, True)
    frameCount = 0
    numero_imagem = 0
    com_placa = False
    cria_trackbars()
    start = False
    while True:
        l_h = cv2.getTrackbarPos("L-H", "TrackbarsHSV")
        l_s = cv2.getTrackbarPos("L-S", "TrackbarsHSV")
        l_v = cv2.getTrackbarPos("L-V", "TrackbarsHSV")
        u_h = cv2.getTrackbarPos("U-H", "TrackbarsHSV")
        u_s = cv2.getTrackbarPos("U-S", "TrackbarsHSV")
        u_v = cv2.getTrackbarPos("U-V", "TrackbarsHSV")
        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])



        x_begin,y_begin,x_end,y_end,x_b_captura,y_b_captura,x_e_captura,y_e_captura = captura_coordenadas()
        oficial_frame = captura_tela(x_begin,y_begin,x_end,y_end,x_b_captura,y_b_captura,x_e_captura,y_e_captura)
        regiao_analise = oficial_frame[y_b_captura:y_b_captura+y_e_captura,x_b_captura:x_b_captura+x_e_captura]
        # Get the foreground mask
        fgmask = fgbg.apply(regiao_analise)
        # Count all the non zero pixels within the mask
        count = np.count_nonzero(fgmask)
        exibicao = cv2.resize(oficial_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("exibicao", exibicao)
        cv2.moveWindow("exibicao", 1367, 0)
        cv2.imshow("regiao_analise", regiao_analise)
        key = cv2.waitKey(1)
        if (frameCount == 1 and count > 1000 and start ==True):
            print('Movimento!')
            cv2.imwrite("./fotos/novos_psavedImage_{}.jpg".format(numero_imagem), oficial_frame)
            numero_imagem = numero_imagem+1
            try:
                image, x, y, w, h, sem_placa = objectDetector(oficial_frame)
                print("Placa detectada!")
                Cropped = image[y:y + h, x:x + w]

                hsv = cv2.cvtColor(Cropped, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, lower_red, upper_red)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel)
                kernel1 = np.ones((25, 25), np.uint8)
                mask = cv2.dilate(mask, kernel1, iterations=1)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 3000:
                        continue

                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    #cv2.drawContours(Cropped, [box], 0, (0, 0, 255), 1)
                primeiro = box[2][0]
                segundo = box[2][1]
                terceiro = box[3][0]
                quarto = box[3][1]
                quinto = box[1][0]
                sexto = box[1][1]
                setimo = box[0][0]
                oitavo = box[0][1]
                pts1 = np.float32([[primeiro, segundo],
                                   [terceiro, quarto],
                                   [quinto, sexto],
                                   [setimo, oitavo]])
                largura_license_plate = box[0][0]
                altura_license_plate = box[0][1]
                pts2 = np.float32([[0, 0],
                                   [largura_license_plate, 0],
                                   [0, altura_license_plate],
                                   [largura_license_plate, altura_license_plate]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                Cropped_perspectiva = cv2.warpPerspective(Cropped, matrix, (largura_license_plate, altura_license_plate))
                Cropped_perspectiva = cv2.resize(Cropped_perspectiva, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
                Cropped_perspectiva_Gray = cv2.cvtColor(Cropped_perspectiva, cv2.COLOR_RGB2GRAY)

                (thresh, Cropped_perspectiva_Gray_bin) = cv2.threshold(Cropped_perspectiva_Gray, 55, 255, cv2.THRESH_BINARY)
                Cropped_perspectiva_Gray_bin = Cropped_perspectiva_Gray_bin[
                                                       40:Cropped_perspectiva_Gray_bin.shape[0] -23,
                                                       16:Cropped_perspectiva_Gray_bin.shape[1] -17 ]
                Cropped_perspectiva_Gray_bin_rotacao = rotate_image(Cropped_perspectiva_Gray_bin, -3)
                #kernel = np.ones((4, 4), np.uint8)
                #result = cv2.erode(result, kernel, iterations=1)
                #result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel=kernel)
                #result = ~result
                #result = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel=kernel)
                #cv2.imshow("Antes", cv2.resize(Cropped_perspectiva, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
                Cropped_perspectiva_Gray_bin_rotacao = cv2.resize(Cropped_perspectiva_Gray_bin_rotacao, None, fx=2, fy=1.5, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Depois", Cropped_perspectiva_Gray_bin_rotacao)
                #cv2.imshow("mask", mask)
                cv2.imwrite("./fotos/placas02/placa{}.jpg".format(numero_imagem), Cropped_perspectiva_Gray_bin_rotacao)
                #cv2.moveWindow("Antes", 1367 + ((oficial_frame.shape)[1]) + 85-300, ((oficial_frame.shape)[0]) - 51)
                #cv2.moveWindow("Depois", 1367 + ((oficial_frame.shape)[1]) + 85-300, ((oficial_frame.shape)[0]) +55+((Cropped_perspectiva_Gray_rotacao_bin.shape)[0]))
                cv2.waitKey(1)

                try:
                    text = pyt.image_to_string(Cropped_perspectiva_Gray_bin_rotacao, lang='eng', config=' --psm 13 ')
                    #text1 = pyt.image_to_string(Cropped_perspectiva_Gray_rotacao_bin, lang='fefont', config=' --psm 13')
                    #text = text.replace(')', '')
                    #text = text.replace(',', '')
                    #text = text.replace("'", '')
                    #text = text.replace(" ", '')
                    # text = pyt.image_to_string(result, lang='eng',config=' --psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    novo_texto = list()
                    text = text.upper()
                    caracteres = digits + ascii_uppercase
                    for letra in text:
                        if letra in (caracteres):
                            novo_texto.append(letra)
                    print("PLACA1:" + text)
                    novo_texto = str(novo_texto)
                    novo_texto = novo_texto.replace(',','')
                    novo_texto = novo_texto.replace('[','')
                    novo_texto = novo_texto.replace(']','')
                    novo_texto = novo_texto.replace("'",'')
                    novo_texto = novo_texto.replace(" ",'')
                    print("PLACA_LIMPA:" , novo_texto)
                    #print("PLACA2:" + text1)

                except  Exception as e:
                    print(e)
                    pass

            except Exception as e:
                print("Placa não detectada {}".format(e))

        if frameCount == 0:
            frameCount = frameCount + 1
        if key != -1:
            print(key)
        if key == 115:
            print("START")
            start = True
        if key == 112:
            print("PAUSE")
            start = False
        if key == 27:
            print("CLOSE")
            break
    cv2.destroyAllWindows()