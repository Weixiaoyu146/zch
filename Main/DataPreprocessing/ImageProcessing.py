import pydicom
import numpy as np
import cv2 as cv
import os

def dicom_to_opencv_gray(dicom_image_path, image_dictory, id):
    dicom_image = pydicom.dcmread(dicom_image_path)
    binary_image = dicom_image.pixel_array
    anterior_binary_image = binary_image[0]
    posterior_binary_image = binary_image[1]
    if "ANTERIOR" in dicom_image_path:
        for file in os.listdir(dicom_image_path[:-19]):
            if "POST" in file:
                dicom_image_path_p = dicom_image_path[:-18] + file
                break
        dicom_image_p = pydicom.dcmread(dicom_image_path_p)
        binary_image_p = dicom_image_p.pixel_array
        anterior_binary_image = binary_image
        posterior_binary_image = binary_image_p
        # cv.imshow(np.uint8(anterior_binary_image))
        # cv.imshow(np.uint8(posterior_binary_image))
    anterior_image = np.uint8(anterior_binary_image)
    posterior_image = np.uint8(posterior_binary_image)

    anterior_image_path = image_dictory + r'/OriginalImages/' + str(id) + 'A.jpg'
    posterior_image_path = image_dictory + r'/OriginalImages/' + str(id) + 'P.jpg'
    # if not os.path.isdir(image_dictory + r'/Image'):
    #     os.makedirs(image_dictory + r'./Image')
    #     os.mkdir(image_dictory + r'/Image' + r'./Original')
    # cv.imencode('.jpg', anterior_image)[1].tofile(anterior_image_path)
    # cv.imencode('.jpg', posterior_image)[1].tofile(posterior_image_path)
    cv.imwrite(anterior_image_path, anterior_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(posterior_image_path, posterior_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    return anterior_image, posterior_image

def contrast_brightness_image(img, alpha, beta):    #第2个参数rario为对比度  第3个参数b为亮度
    blank = np.zeros(img.shape, img.dtype)
    dst = cv.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst

def threshold_demo_gray(image):
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(image, 0, 255, cv.THRESH_TOZERO | cv.THRESH_TRIANGLE)
    return binary

def median_blur_demo(image):
    dst = cv.medianBlur(image, 3)
    return dst

def cut_wholebody(image):
    cut_image = cv.erode(src=image, kernel=np.ones((3, 3), np.uint8), iterations=1)
    height, width = cut_image.shape
    break_flag = False
    for row in range(height):
        for col in range(width):
            if cut_image[row, col] > 1:
                minY = row
                break_flag = True
                break
        if break_flag == True:
            break

    break_flag = False
    for row in range(height):
        for col in range(width):
            if cut_image[height-row-1, col] > 1:
                maxY =height-row-1
                break_flag = True
                break
        if break_flag == True:
            break

    break_flag = False
    for col in range(width):
        for row in range(height):
            if cut_image[row, col] > 1:
                minX = col
                break_flag = True
                break
        if break_flag == True:
            break

    break_flag = False
    for col in range(width):
        for row in range(height):
            if cut_image[row, width-col-1] > 1:
                maxX = width-col-1
                break_flag = True
                break
        if break_flag == True:
            break

    cut_image = image[minY:maxY, minX:maxX]
    return minX, maxX, minY, maxY, cut_image

def preprocessing(image):
    # image = contrast_brightness_image(image, 3.9 / cv.mean(image)[0], 0)
    binary_image = threshold_demo_gray(image)
    kernel = np.ones((3, 3), np.uint8)
    cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel, binary_image, iterations=1)  # 闭运算
    binary_image = median_blur_demo(binary_image)  # 中值滤波
    cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel, binary_image, iterations=1)  # 开运算
    mask = cv.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=binary_image)  # 掩码运算
    # 均衡化之前进行高斯模糊确保像素值分布足够宽泛
    cv.GaussianBlur(src=mask, ksize=(5, 5), dst=mask, sigmaX=0.75, sigmaY=0.75)
    cv.equalizeHist(mask, mask)
    return image, mask

def cut_preprocessing_image(original_anterior_image, original_posterior_image, image_dictory, id):
    # 预处理
    anterior_image, anterior_image_mask = preprocessing(original_anterior_image)
    posterior_image, posterior_image_mask = preprocessing(original_posterior_image)
    # 采用简单阈值分割代替模糊集理论阈值分割
    minX, maxX, minY, maxY, anterior_cut_image_mask = cut_wholebody(anterior_image_mask)  # 将图片裁剪到保留全身的最小矩形区域。
    anterior_cut_image = anterior_image[minY:maxY, minX:maxX]
    cv.threshold(anterior_cut_image_mask, 150, 255, cv.THRESH_TOZERO, anterior_cut_image_mask)  # 简单阈值分割
    cv.morphologyEx(anterior_cut_image_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), anterior_cut_image_mask, iterations=1)  # 开运算消除孤立像素点

    # 为了和前身图保持一致尺寸 采用镜像切割已保持整体的一致性
    posterior_cut_image_mask = posterior_image_mask[minY:maxY, original_posterior_image.shape[1] - maxX:original_posterior_image.shape[1] - minX]
    posterior_cut_image = posterior_image[minY:maxY, original_posterior_image.shape[1] - maxX:original_posterior_image.shape[1] - minX]
    cv.threshold(posterior_cut_image_mask, 150, 255, cv.THRESH_TOZERO, posterior_cut_image_mask)  # 简单阈值分割
    cv.morphologyEx(posterior_cut_image_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), posterior_cut_image_mask, iterations=1)  # 开运算消除孤立像素点

    anterior_image_path = image_dictory + r'/DenoisedImages/' + str(id) + 'a.jpg'
    posterior_image_path = image_dictory + r'/DenoisedImages/' + str(id) + 'p.jpg'
    anterior_image_mask_path = image_dictory + r'/MaskImages/' + str(id) + 'a.jpg'
    posterior_image_mask_path = image_dictory + r'/MaskImages/' + str(id) + 'p.jpg'

    anterior_cut_image_white = ~anterior_cut_image
    posterior_cut_image_white = ~posterior_cut_image
    anterior_image_white_path = image_dictory + r'/CroppedWhiteBackgroundImages/' + str(id) + 'a.jpg'
    posterior_image_white_path = image_dictory + r'/CroppedWhiteBackgroundImages/' + str(id) + 'p.jpg'

    # if not os.path.isdir(image_dictory + r'/Image/Preprocessed'):
    #     os.mkdir(image_dictory + r'/Image' + r'./Preprocessed')
    cv.imwrite(anterior_image_path, anterior_cut_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(posterior_image_path, posterior_cut_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(anterior_image_mask_path, anterior_cut_image_mask, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(posterior_image_mask_path, posterior_cut_image_mask, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imwrite(anterior_image_white_path, anterior_cut_image_white, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imwrite(posterior_image_white_path, posterior_cut_image_white, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imencode('.jpg', anterior_cut_image)[1].tofile(anterior_image_path)
    # cv.imencode('.jpg', posterior_cut_image)[1].tofile(posterior_image_path)
    # cv.imencode('.jpg', anterior_cut_image_mask)[1].tofile(anterior_image_mask_path)
    # cv.imencode('.jpg', posterior_cut_image_mask)[1].tofile(posterior_image_mask_path)

    # cv.imencode('.jpg', anterior_cut_image_white)[1].tofile(anterior_image_white_path)
    # cv.imencode('.jpg', posterior_cut_image_white)[1].tofile(posterior_image_white_path)

    return anterior_cut_image, posterior_cut_image, anterior_cut_image_mask, posterior_cut_image_mask

def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),\
                    Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-2, 0), Point(-2, 1),\
                    Point(-2, 2), Point(-2, -1), Point(-2, -2), Point(2, 0), Point(2, 1),\
                    Point(2, 2), Point(2, -1), Point(2, -2), Point(1, -2), Point(1, 2),\
                    Point(0, -2), Point(0, 2), Point(-1, -2), Point(-1, 2)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img,seeds,max,p = 1):
    height, weight = img.shape
    seedMark = np.ones(img.shape)
    seedList = []
    regionGrow_area = []
    for seed in seeds:
        seedList.append(seed)
    label = 0
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x,currentPoint.y] = label
        regionGrow_area.append((currentPoint.y, currentPoint.x))
        for i in range(24):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            gray_Value = int(img[tmpX,tmpY])
            #grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if gray_Value > max*0.6 and seedMark[tmpX,tmpY] == 1:
                seedMark[tmpX,tmpY] = label
                regionGrow_area.append((tmpY, tmpX))
                seedList.append(Point(tmpX,tmpY))
    return regionGrow_area

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

class Segmentation:
    a = b = c = d = e = f = g = h = i = j = k = l = m = n = o = p = q = r = s = t = (0, 0)
    head_left = head_right = foot_joint_row = foot_row = left_hand_row =right_hand_row = 0
    vertbra_contour = []
    vertbra = []
    kidneys = []
    bladder = []
    imageAP = imagePA = []        #预处理后分割用图像
    image_height = image_width = 0    #图像高宽

    image_histogram_AP = image_histogram_PA = []     #前身图直方图列表
#后身图直方图列表
    OriginalAP = ''            #前身图
    OriginalPA = ''            #后身图
    imageAP_name = ''           #前身图名字
    imagePA_name = ''           #后身图名字
    Part_Image = []         #全部分支 0:前身 1：后身

    #类的实例化
    def __init__(self, imageAP, imagePA, OriginalAP, OriginalPA, image_dictory = ''):
        if len(imageAP.shape) != 2:
            print("error image struct")
            return
        self.image_histogram_AP = np.sum(imageAP, axis=1)/imageAP.shape[1]
        self.image_histogram_PA = np.sum(imagePA, axis=1)/imagePA.shape[1]

        self.OriginalAP = OriginalAP
        self.OriginalPA = OriginalPA
        self.image_dictory = image_dictory
        self.imageAP_name = image_dictory + r'/Image/Preprocessed/anterior_image_mask.jpg'
        self.imagePA_name = image_dictory + r'/Image/Preprocessed/posterior_image_mask.jpg'
        self.imageAP = imageAP
        self.imagePA = imagePA
        self.image_height, self.image_width = imageAP.shape

    #进行点的定位和分割
    def Segment(self):

        #点的定位以及脊柱的定位和肾脏的定位
        self.get_ab()
        self.get_ij()
        self.get_cdefgh()
        self.get_kl()
        self.get_mn()
        self.get_opst()
        self.get_joint()
        self.get_vertbra_contour()
        self.get_vertbra_kidneys()
        #self.show_segment()
        #将分割好的图片按顺序入队
        #序列顺序为：0：头部 1：左肩 2：右肩： 3：左胸腔 4：右胸腔 5：左手关节 6：右手关节
        #7：脊柱 8：骨盆 9：左膝关节 10：右膝关节 11：左踝关节 12：右踝关节
        temp_list_0 = []    #前身序列
        temp_list_1 = []    #后身序列
        #获取分割图像
        if not os.path.isdir(self.image_dictory + r'/Image/Segmentation/PartImage'):
            os.mkdir(self.image_dictory + r'/Image/Segmentation' + r'./PartImage')
        ap_0, pa_0 = self.cut_head()
        ap_1, pa_1, ap_2, pa_2 = self.cut_shol()
        ap_3, pa_3, ap_4, pa_4 = self.cut_chest()
        ap_5, pa_5, ap_6, pa_6 = self.cut_elbow_joint()
        ap_7, pa_7 = self.cut_vertbra()
        ap_8, pa_8 = self.cut_pelvis()
        ap_9, pa_9, ap_10, pa_10 = self.cut_knee_joint()
        ap_11, pa_11, ap_12, pa_12 = self.cut_ankle_joint()
        #分割图像剪切
        temp_list_0 = self.Cut_Region(ap_0, ap_1, ap_2, ap_3, ap_4, ap_5, ap_6, ap_7, ap_8, ap_9, ap_10, ap_11, ap_12)
        temp_list_1 = self.Cut_Region(pa_0, pa_1, pa_2, pa_3, pa_4, pa_5, pa_6, pa_7, pa_8, pa_9, pa_10, pa_11, pa_12)
        temp_list_0 = self.Image_Resize(temp_list_0)
        temp_list_1 = self.Image_Resize(temp_list_1)

        Part_Image = []
        Part_Image.append(temp_list_0)
        Part_Image.append(temp_list_1)

        self.Part_Image = Part_Image.copy()

        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_head.jpg', self.Part_Image[0][0], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_shoulder.jpg', self.Part_Image[0][1], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_shoulder.jpg', self.Part_Image[0][2], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_chest.jpg', self.Part_Image[0][3], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_chest.jpg', self.Part_Image[0][4], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_elbow.jpg', self.Part_Image[0][5], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_elbow.jpg', self.Part_Image[0][6], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_vertbra.jpg', self.Part_Image[0][7], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_pelvis.jpg', self.Part_Image[0][8], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_knee.jpg', self.Part_Image[0][9], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_knee.jpg', self.Part_Image[0][10], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_ankle.jpg', self.Part_Image[0][11], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_ankle.jpg', self.Part_Image[0][12], [int(cv.IMWRITE_JPEG_QUALITY), 100])

        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_head.jpg', self.Part_Image[1][0], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_shoulder.jpg', self.Part_Image[1][1], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_shoulder.jpg', self.Part_Image[1][2], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_chest.jpg', self.Part_Image[1][3], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_chest.jpg', self.Part_Image[1][4], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_elbow.jpg', self.Part_Image[1][5], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_elbow.jpg', self.Part_Image[1][6], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_vertbra.jpg', self.Part_Image[1][7], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_pelvis.jpg', self.Part_Image[1][8], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_knee.jpg', self.Part_Image[1][9], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_knee.jpg', self.Part_Image[1][10], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_ankle.jpg', self.Part_Image[1][11], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_ankle.jpg', self.Part_Image[1][12], [int(cv.IMWRITE_JPEG_QUALITY), 100])

        # img_channels = 3
        # img_rows = 87
        # img_cols = 59
        # nb_classes = 2
        #
        # model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
        # model.load_weights(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model_Test\0-0-ResNet50.h5')

        predict = []

        # ap_0_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-0-ResNet50.h5')
        # ap_0_images = []
        # ap_0_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_head.jpg')
        # ap_0_images.append(ap_0_image)
        # ap_0_images = np.array(ap_0_images)
        # ap_0_images = ap_0_images.astype('float32')
        # ap_0_images /= 128.
        # ap_0_predict = ap_0_model.predict(ap_0_images)
        # predict.append(ap_0_predict[0])
        #
        # ap_1_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-1-ResNet50.h5')
        # ap_1_images = []
        # ap_1_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_shoulder.jpg')
        # ap_1_images.append(ap_1_image)
        # ap_1_images = np.array(ap_1_images)
        # ap_1_images = ap_1_images.astype('float32')
        # ap_1_images /= 128.
        # ap_1_predict = ap_1_model.predict(ap_1_images)
        # predict.append(ap_1_predict[0])
        #
        # ap_2_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-2-ResNet50.h5')
        # ap_2_images = []
        # ap_2_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_shoulder.jpg')
        # ap_2_images.append(ap_2_image)
        # ap_2_images = np.array(ap_2_images)
        # ap_2_images = ap_2_images.astype('float32')
        # ap_2_images /= 128.
        # ap_2_predict = ap_2_model.predict(ap_2_images)
        # predict.append(ap_2_predict[0])
        #
        # ap_3_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-3-ResNet50.h5')
        # ap_3_images = []
        # ap_3_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_chest.jpg')
        # ap_3_images.append(ap_3_image)
        # ap_3_images = np.array(ap_3_images)
        # ap_3_images = ap_3_images.astype('float32')
        # ap_3_images /= 128.
        # ap_3_predict = ap_3_model.predict(ap_3_images)
        # predict.append(ap_3_predict[0])
        #
        # ap_4_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-4-ResNet50.h5')
        # ap_4_images = []
        # ap_4_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_chest.jpg')
        # ap_4_images.append(ap_4_image)
        # ap_4_images = np.array(ap_4_images)
        # ap_4_images = ap_4_images.astype('float32')
        # ap_4_images /= 128.
        # ap_4_predict = ap_4_model.predict(ap_4_images)
        # predict.append(ap_4_predict[0])
        #
        # ap_5_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-5-ResNet50.h5')
        # ap_5_images = []
        # ap_5_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_elbow.jpg')
        # ap_5_images.append(ap_5_image)
        # ap_5_images = np.array(ap_5_images)
        # ap_5_images = ap_5_images.astype('float32')
        # ap_5_images /= 128.
        # ap_5_predict = ap_5_model.predict(ap_5_images)
        # predict.append(ap_5_predict[0])
        #
        # ap_6_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-6-ResNet50.h5')
        # ap_6_images = []
        # ap_6_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_elbow.jpg')
        # ap_6_images.append(ap_6_image)
        # ap_6_images = np.array(ap_6_images)
        # ap_6_images = ap_6_images.astype('float32')
        # ap_6_images /= 128.
        # ap_6_predict = ap_6_model.predict(ap_6_images)
        # predict.append(ap_6_predict[0])
        #
        # ap_7_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-7-ResNet50.h5')
        # ap_7_images = []
        # ap_7_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_vertbra.jpg')
        # ap_7_images.append(ap_7_image)
        # ap_7_images = np.array(ap_7_images)
        # ap_7_images = ap_7_images.astype('float32')
        # ap_7_images /= 128.
        # ap_7_predict = ap_7_model.predict(ap_7_images)
        # predict.append(ap_7_predict[0])
        #
        # ap_8_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-8-ResNet50.h5')
        # ap_8_images = []
        # ap_8_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_pelvis.jpg')
        # ap_8_images.append(ap_8_image)
        # ap_8_images = np.array(ap_8_images)
        # ap_8_images = ap_8_images.astype('float32')
        # ap_8_images /= 128.
        # ap_8_predict = ap_8_model.predict(ap_8_images)
        # predict.append(ap_8_predict[0])
        #
        # ap_9_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-9-ResNet50.h5')
        # ap_9_images = []
        # ap_9_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_knee.jpg')
        # ap_9_images.append(ap_9_image)
        # ap_9_images = np.array(ap_9_images)
        # ap_9_images = ap_9_images.astype('float32')
        # ap_9_images /= 128.
        # ap_9_predict = ap_9_model.predict(ap_9_images)
        # predict.append(ap_9_predict[0])
        #
        # ap_10_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-10-ResNet50.h5')
        # ap_10_images = []
        # ap_10_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_knee.jpg')
        # ap_10_images.append(ap_10_image)
        # ap_10_images = np.array(ap_10_images)
        # ap_10_images = ap_10_images.astype('float32')
        # ap_10_images /= 128.
        # ap_10_predict = ap_10_model.predict(ap_10_images)
        # predict.append(ap_10_predict[0])
        #
        # ap_11_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-11-ResNet50.h5')
        # ap_11_images = []
        # ap_11_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_left_ankle.jpg')
        # ap_11_images.append(ap_11_image)
        # ap_11_images = np.array(ap_11_images)
        # ap_11_images = ap_11_images.astype('float32')
        # ap_11_images /= 128.
        # ap_11_predict = ap_11_model.predict(ap_11_images)
        # predict.append(ap_11_predict[0])
        #
        # ap_12_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-12-ResNet50.h5')
        # ap_12_images = []
        # ap_12_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_right_ankle.jpg')
        # ap_12_images.append(ap_12_image)
        # ap_12_images = np.array(ap_12_images)
        # ap_12_images = ap_12_images.astype('float32')
        # ap_12_images /= 128.
        # ap_12_predict = ap_12_model.predict(ap_12_images)
        # predict.append(ap_12_predict[0])
        #
        # pa_0_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-0-ResNet50.h5')
        # pa_0_images = []
        # pa_0_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_head.jpg')
        # pa_0_images.append(pa_0_image)
        # pa_0_images = np.array(pa_0_images)
        # pa_0_images = pa_0_images.astype('float32')
        # pa_0_images /= 128.
        # pa_0_predict = pa_0_model.predict(pa_0_images)
        # predict.append(pa_0_predict[0])
        #
        # pa_1_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-1-ResNet50.h5')
        # pa_1_images = []
        # pa_1_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_shoulder.jpg')
        # pa_1_images.append(pa_1_image)
        # pa_1_images = np.array(pa_1_images)
        # pa_1_images = pa_1_images.astype('float32')
        # pa_1_images /= 128.
        # pa_1_predict = pa_1_model.predict(pa_1_images)
        # predict.append(pa_1_predict[0])
        #
        # pa_2_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-2-ResNet50.h5')
        # pa_2_images = []
        # pa_2_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_shoulder.jpg')
        # pa_2_images.append(pa_2_image)
        # pa_2_images = np.array(pa_2_images)
        # pa_2_images = pa_2_images.astype('float32')
        # pa_2_images /= 128.
        # pa_2_predict = pa_2_model.predict(pa_2_images)
        # predict.append(pa_2_predict[0])
        #
        # pa_3_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-3-ResNet50.h5')
        # pa_3_images = []
        # pa_3_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_chest.jpg')
        # pa_3_images.append(pa_3_image)
        # pa_3_images = np.array(pa_3_images)
        # pa_3_images = pa_3_images.astype('float32')
        # pa_3_images /= 128.
        # pa_3_predict = pa_3_model.predict(pa_3_images)
        # predict.append(pa_3_predict[0])
        #
        # pa_4_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-4-ResNet50.h5')
        # pa_4_images = []
        # pa_4_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_chest.jpg')
        # pa_4_images.append(pa_4_image)
        # pa_4_images = np.array(pa_4_images)
        # pa_4_images = pa_4_images.astype('float32')
        # pa_4_images /= 128.
        # pa_4_predict = pa_4_model.predict(pa_4_images)
        # predict.append(pa_4_predict[0])
        #
        # pa_5_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-5-ResNet50.h5')
        # pa_5_images = []
        # pa_5_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_elbow.jpg')
        # pa_5_images.append(pa_5_image)
        # pa_5_images = np.array(pa_5_images)
        # pa_5_images = pa_5_images.astype('float32')
        # pa_5_images /= 128.
        # pa_5_predict = pa_5_model.predict(pa_5_images)
        # predict.append(pa_5_predict[0])
        #
        # pa_6_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-6-ResNet50.h5')
        # pa_6_images = []
        # pa_6_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_elbow.jpg')
        # pa_6_images.append(pa_6_image)
        # pa_6_images = np.array(pa_6_images)
        # pa_6_images = pa_6_images.astype('float32')
        # pa_6_images /= 128.
        # pa_6_predict = pa_6_model.predict(pa_6_images)
        # predict.append(pa_6_predict[0])
        #
        # pa_7_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-7-ResNet50.h5')
        # pa_7_images = []
        # pa_7_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_vertbra.jpg')
        # pa_7_images.append(pa_7_image)
        # pa_7_images = np.array(pa_7_images)
        # pa_7_images = pa_7_images.astype('float32')
        # pa_7_images /= 128.
        # pa_7_predict = pa_7_model.predict(pa_7_images)
        # predict.append(pa_7_predict[0])
        #
        # pa_8_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-8-ResNet50.h5')
        # pa_8_images = []
        # pa_8_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_pelvis.jpg')
        # pa_8_images.append(pa_8_image)
        # pa_8_images = np.array(pa_8_images)
        # pa_8_images = pa_8_images.astype('float32')
        # pa_8_images /= 128.
        # pa_8_predict = pa_8_model.predict(pa_8_images)
        # predict.append(pa_8_predict[0])
        #
        # pa_9_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-9-ResNet50.h5')
        # pa_9_images = []
        # pa_9_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_knee.jpg')
        # pa_9_images.append(pa_9_image)
        # pa_9_images = np.array(pa_9_images)
        # pa_9_images = pa_9_images.astype('float32')
        # pa_9_images /= 128.
        # pa_9_predict = pa_9_model.predict(pa_9_images)
        # predict.append(pa_9_predict[0])
        #
        # pa_10_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-10-ResNet50.h5')
        # pa_10_images = []
        # pa_10_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_knee.jpg')
        # pa_10_images.append(pa_10_image)
        # pa_10_images = np.array(pa_10_images)
        # pa_10_images = pa_10_images.astype('float32')
        # pa_10_images /= 128.
        # pa_10_predict = pa_10_model.predict(pa_10_images)
        # predict.append(pa_10_predict[0])
        #
        # pa_11_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-11-ResNet50.h5')
        # pa_11_images = []
        # pa_11_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_left_ankle.jpg')
        # pa_11_images.append(pa_11_image)
        # pa_11_images = np.array(pa_11_images)
        # pa_11_images = pa_11_images.astype('float32')
        # pa_11_images /= 128.
        # pa_11_predict = pa_11_model.predict(pa_11_images)
        # predict.append(pa_11_predict[0])
        #
        # pa_12_model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\1-12-ResNet50.h5')
        # pa_12_images = []
        # pa_12_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/posterior_right_ankle.jpg')
        # pa_12_images.append(pa_12_image)
        # pa_12_images = pa.array(pa_12_images)
        # pa_12_images = pa_12_images.astype('float32')
        # pa_12_images /= 128.
        # pa_12_predict = pa_12_model.predict(pa_12_images)
        # predict.append(pa_12_predict[0])

        # print(predict)

        # model_image = cv.imread(self.image_dictory + r'/Image/Segmentation/PartImage/anterior_head.jpg')
        # model = load_model(r'D:\Documents\PycharmProjects\BoneScan-QT1\Model\0-0-ResNet50.h5')
        # model_images = []
        # model_images.append(model_image)
        # model_images = np.array(model_images)
        # model_images = model_images.astype('float32')
        # model_images /= 128.
        # predict = model.predict(model_images)

        #image_cluster_list = self.Region_Detect(pa_8)
        #self.Feature_Detect(image_cluster_list, pa_8)
        self.show_point()
        self.show_segment()
        # cv.waitKey(10)
        return predict

    #在图像中显示点和划线
    def show_point(self):
        imagePA = self.imagePA
        imagePA = cv.cvtColor(imagePA, cv.COLOR_GRAY2RGB)
        imageAP = self.imageAP
        imageAP = cv.cvtColor(imageAP, cv.COLOR_GRAY2RGB)
        cv.circle(img=imagePA, center=self.a, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.b, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.c, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.d, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.e, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.f, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.g, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.h, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.i, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.j, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.k, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.l, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.m, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.n, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.o[0] - 1, self.o[1]), radius=3, color=(0, 255, 0), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.p[0] - 1, self.p[1]), radius=3, color=(0, 255, 0), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.s[0] - 1, self.s[1]), radius=3, color=(0, 255, 0), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.t[0] - 1, self.t[1]), radius=3, color=(0, 255, 0), thickness=3)

        cv.circle(img=imageAP, center=self.o, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageAP, center=self.p, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageAP, center=self.s, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageAP, center=self.t, radius=3, color=(0, 0, 255), thickness=3)



        # print("a = ", self.a)
        # print("b = ", self.b)
        # print("c = ", self.c)
        # print("d = ", self.d)
        # print("e = ", self.e)
        # print("f = ", self.f)
        # print("g = ", self.g)
        # print("h = ", self.h)
        # print("i = ", self.i)
        # print("j = ", self.j)
        # print("k = ", self.k)
        # print("l = ", self.l)

        # if not os.path.isfile(self.image_dictory + r'/Image/Segmentation/posterior_image.jpg'):
        #     if not os.path.isdir(self.image_dictory + r'/Image/Segmentation'):
        #         os.mkdir(self.image_dictory + r'/Image' + r'./Segmentation')
        #     cv.imwrite(self.image_dictory + r'/Image/Segmentation/posterior_image.jpg', imagePA, [int(cv.IMWRITE_JPEG_QUALITY), 95])

        imageAP = self.imageAP

    #1.在图中绘制方框 2.分割部位分别保存
    def show_segment(self):
        #1.绘制边框直线
        imagePA = self.OriginalPA
        imagePA = cv.cvtColor(imagePA, cv.COLOR_GRAY2RGB)
        imageAP = self.OriginalAP
        imageAP = cv.cvtColor(imageAP, cv.COLOR_GRAY2RGB)
        red = (0, 0, 255)
        green = (0, 255, 0)
        black = (0, 0, 0)

        #for point in self.kidneys:
        #    imagePA[point[1], point[0]] = red
        #for point in self.bladder:
        #    imagePA[point[1], point[0]] = red
        for point in self.vertbra:
            imagePA[point[1], point[0]] = red

        '''
        cv.circle(img=imagePA, center=self.a, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.b, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.c, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.d, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.e, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.f, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.g, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.h, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.i, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.j, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.k, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.l, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.m, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=self.n, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.o[0] - 1, self.o[1]), radius=3, color=(0, 255, 0), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.p[0] - 1, self.p[1]), radius=3, color=(0, 255, 0), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.s[0] - 1, self.s[1]), radius=3, color=(0, 255, 0), thickness=3)
        cv.circle(img=imagePA, center=(self.image_width - self.t[0] - 1, self.t[1]), radius=3, color=(0, 255, 0), thickness=3)

        cv.circle(img=imageAP, center=self.o, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageAP, center=self.p, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageAP, center=self.s, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageAP, center=self.t, radius=3, color=(0, 0, 255), thickness=3)
        '''

        #头部
        cv.rectangle(imagePA, (self.head_left, 0), (self.head_right, self.a[1]), red, 1)
        cv.rectangle(imageAP, (self.image_width - 1 - self.head_left, 0), (self.image_width - 1 - self.head_right, self.a[1]), red, 1)
        #左肩
        cv.line(imagePA, (0, self.e[1]), self.e, red, 1)
        cv.line(imageAP, (self.image_width - 1, self.e[1]), (self.image_width - 1 - self.e[0], self.e[1]), red, 1)
        cv.line(imagePA, (0, self.g[1]), self.g, red, 1)
        cv.line(imageAP, (self.image_width - 1, self.g[1]), (self.image_width - 1 - self.g[0], self.g[1]), red, 1)
        cv.line(imagePA, self.e, self.c, red, 1)
        cv.line(imageAP, (self.image_width - 1 - self.e[0], self.e[1]), (self.image_width - 1 - self.c[0], self.c[1]), red, 1)
        cv.line(imagePA, self.g, self.c, red, 1)
        cv.line(imageAP, (self.image_width - 1 - self.g[0], self.g[1]), (self.image_width - 1 - self.c[0], self.c[1]), red, 1)
        #右肩
        cv.line(imagePA, (self.image_width-1, self.f[1]), self.f, red, 1)
        cv.line(imagePA, (self.image_width-1, self.h[1]), self.h, red, 1)
        cv.line(imagePA, self.f, self.d, red, 1)
        cv.line(imagePA, self.d, self.h, red, 1)

        cv.line(imageAP, (0, self.f[1]), (self.image_width - 1 - self.f[0], self.f[1]), red, 1)
        cv.line(imageAP, (0, self.h[1]), (self.image_width - 1 - self.h[0], self.h[1]), red, 1)
        cv.line(imageAP, (self.image_width - 1 - self.f[0], self.f[1]), (self.image_width - 1 - self.d[0], self.d[1]), red, 1)
        cv.line(imageAP, (self.image_width - 1 - self.d[0], self.d[1]), (self.image_width - 1 - self.h[0], self.h[1]), red, 1)

        #左右手
        cv.line(imagePA, self.g, (self.g[0], self.m[1]), red, 1)
        cv.line(imagePA, (self.g[0], self.m[1]), (0, self.m[1]), red, 1)

        cv.line(imageAP, (self.image_width - 1 - self.g[0], self.g[1]), (self.image_width - 1 - self.g[0], self.m[1]), red, 1)
        cv.line(imageAP, (self.image_width - 1 - self.g[0], self.m[1]), (self.image_width - 1, self.m[1]), red, 1)

        cv.line(imagePA, self.h, (self.h[0], self.n[1]), red, 1)
        cv.line(imagePA, (self.image_width-1, self.n[1]), (self.h[0], self.n[1]), red, 1)

        cv.line(imageAP, (self.image_width - 1 - self.h[0], self.h[1]), (self.image_width - 1 - self.h[0], self.n[1]), red, 1)
        cv.line(imageAP, (0, self.n[1]), (self.image_width - 1 - self.h[0], self.n[1]), red, 1)
        #左右手关节
        cv.rectangle(imagePA, (0, self.left_hand_row-20), (self.g[0], self.left_hand_row+20), red, 1)
        cv.rectangle(imagePA, (self.h[0], self.right_hand_row-20), (self.image_width-1, self.right_hand_row+20), red, 1)

        cv.rectangle(imageAP, (self.image_width - 1, self.left_hand_row-20), (self.image_width - 1 - self.g[0], self.left_hand_row+20), red, 1)
        cv.rectangle(imageAP, (self.image_width - 1 - self.h[0], self.right_hand_row-20), (0, self.right_hand_row+20), red, 1)



        #盆骨
        cv.rectangle(imagePA, (self.image_width - self.o[0] - 1, self.i[1]), (self.image_width - self.p[0] - 1, self.n[1]), red, 1)

        cv.rectangle(imageAP, (self.o[0], self.i[1]), (self.p[0], self.n[1]), red, 1)


        #左右腿
        cv.rectangle(imagePA, (self.e[0], self.m[1]), (int((self.e[0]+self.f[0])/2), self.image_height - 1), red, 1)
        cv.rectangle(imagePA, (int((self.e[0] + self.f[0]) / 2), self.m[1]), (self.f[0], self.image_height - 1), red, 1)

        cv.rectangle(imageAP, (self.image_width - 1 - self.e[0], self.m[1]), (self.image_width - 1 - int((self.e[0]+self.f[0])/2), self.image_height - 1), red, 1)
        cv.rectangle(imageAP, (self.image_width - 1 - int((self.e[0] + self.f[0]) / 2), self.m[1]), (self.image_width - 1 - self.f[0], self.image_height - 1), red, 1)
        #左右腿关节

        cv.rectangle(imagePA, (self.e[0], self.foot_joint_row-25), (int((self.e[0]+self.f[0])/2), self.foot_joint_row+25), red, 1)
        cv.rectangle(imagePA, (int((self.e[0]+self.f[0])/2), self.foot_joint_row-25), (self.f[0], self.foot_joint_row+25), red, 1)

        cv.rectangle(imageAP, (self.image_width - 1 - self.e[0], self.foot_joint_row-25), (self.image_width - 1 - int((self.e[0]+self.f[0])/2), self.foot_joint_row+25), red, 1)
        cv.rectangle(imageAP, (self.image_width - 1 - int((self.e[0]+self.f[0])/2), self.foot_joint_row-25), (self.image_width - 1 - self.f[0], self.foot_joint_row+25), red, 1)

        #左右脚掌
        cv.rectangle(imagePA, (self.e[0], self.foot_row-15), (int((self.e[0]+self.f[0])/2), self.image_height), red, 1)
        cv.rectangle(imagePA, (int((self.e[0]+self.f[0])/2), self.foot_row-15), (self.f[0], self.image_height), red, 1)

        cv.rectangle(imageAP, (self.image_width - 1 - self.e[0], self.foot_row-15), (self.image_width - 1 - int((self.e[0]+self.f[0])/2), self.image_height), red, 1)
        cv.rectangle(imageAP, (self.image_width - 1 - int((self.e[0]+self.f[0])/2), self.foot_row-15), (self.image_width - 1 - self.f[0], self.image_height), red, 1)

        #脊柱，胸腔
        # cv.line(imagePA, self.a, self.i, red, 1)
        # cv.line(imagePA, self.b, self.j, red, 1)
        '''
        min_height = self.image_height - 1
        for point in self.vertbra:
            if point[1] < min_height:
                min_height = point[1]

        if min_height > self.a[1]+15:
            cv.line(imagePA, self.a, self.i, red, 1)
            cv.line(imagePA, self.b, self.j, red, 1)
        else:
            for point in self.vertbra:
                imagePA[point[1]][point[0]] = red
        '''


        imagePA_temp = imagePA.copy()
        if not os.path.isdir(self.image_dictory + r'/Image/Segmentation'):
            os.mkdir(self.image_dictory + r'/Image' + r'./Segmentation')
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/posterior_image.jpg', imagePA, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(self.image_dictory + r'/Image/Segmentation/anterior_image.jpg', imageAP, [int(cv.IMWRITE_JPEG_QUALITY), 100])











        return imagePA

    #debug函数 用来测试全身图比例
    def get_line(self, Y_height, *Y_heights):
        image = self.imagePA
        for col in range(image.shape[1]): #图像宽度
            image[Y_height][col] = 255
            for Y_h in Y_heights:
                image[Y_h][col] = 255
        cv.imshow(self.imagePA_name, image)
        cv.waitKey(500)
        # debug函数 用来测试全身图比例
    def get_line_X(self, X_height, *X_heights):
        image = self.imagePA
        for row in range(image.shape[0]):  # 图像宽度
            image[row][X_height] = 255
            for X in X_heights:
                image[row][X] = 255
        cv.imshow(self.imagePA_name, image)
        cv.waitKey(500)
    #定位ab两点以及头部位置
    def get_ab(self):
        Y_10 = int(0.10 * self.image_height)
        Y_20 = int(0.20 * self.image_height)
        min = 256
        index = 0
        X_left = 0
        X_right = 0
        #寻找ab所在行
        for row in range(Y_10, Y_20):
            if self.image_histogram_PA[row] < min:
                min = self.image_histogram_PA[row]
                index = row
        #寻找ab所在列
        for col in range(self.image_width):
            if self.imagePA[index][col] != 0:
                X_left = col
                break
        for col in range(self.image_width):
            if self.imagePA[index][self.image_width-col-1] != 0:
                X_right =self.image_width-col-1
                break
        self.a = (X_left, index)
        self.b = (X_right, index)
        if abs(X_right-X_left) == 0:
            self.b = (X_right+10, index)
            if self.a[0] > 10:
                self.a = (X_left-10, index)
        head_left = self.image_width - 1
        head_right = self.image_width - 1

        #大脑左右边界无法通过特征点来确定，需要初始定位：
        first_nozeros, div_first_nozeros = self.get_first_nozeros(image=self.imagePA, Y_1=0, Y_2=self.a[1])
        for row in range(self.a[1]):
            if first_nozeros[row] < head_left and first_nozeros[row] != 0:
                head_left = first_nozeros[row]

        min = self.image_width - 1
        for row in range(self.a[1]):
            if div_first_nozeros[row] < min and div_first_nozeros[row] != 0:
                min = div_first_nozeros[row]
                head_right = self.image_width - div_first_nozeros[row] - 1
        self.head_left = head_left
        self.head_right = head_right
    #定位i,j两点位置并依据ab位置修正
    def get_ij(self):
        Y_30 = int(0.30 * self.image_height)
        Y_45 = int(0.45 * self.image_height)
        min = 256
        index = 0
        X_left = 0
        X_right = 0
        # 寻找ij所在行
        for row in range(Y_30, Y_45):
            if self.image_histogram_PA[row] < min:
                min = self.image_histogram_PA[row]
                index = row
        # 寻找ij所在列
        for col in range(int(self.image_width/4), self.image_width):
            if self.imagePA[index][col] != 0:
                X_left = col
                break
        for col in range(int(self.image_width/4), self.image_width):
            if self.imagePA[index][self.image_width - col - 1] != 0:
                X_right = self.image_width - col - 1
                break
        self.i = (X_left, index)
        self.j = (X_right, index)
        #i,j两点X坐标修正：
        X_ab = float(self.b[0] - self.a[0])
        X_ij = float(self.j[0] - self.i[0])
        X_ai = float(self.a[0] - self.i[0])
        X_bj = float(self.b[0] - self.j[0])
        if X_ij/X_ab > 1.2:
            if abs(X_ai)-abs(X_bj) > 15:
                X_left = self.a[0] + self.j[0] - self.b[0]
                self.i = (X_left, index)
            if abs(X_bj)-abs(X_ai) > 15:
                X_right = self.b[0] + self.i[0] - self.a[0]
                self.j = (X_right, index)
    #定位c,d,e,f,g,h位置
    def get_cdefgh(self):
        #cdefgh定位坐标参数：
        Y_ce = 0
        Y_g = 0
        Y_df = 0
        Y_h = 0
        X_eg = 0
        X_c = 0
        X_fh = 0
        X_d = 0
        Y_1 = self.a[1]
        Y_2 = int(0.20 * self.image_height)
        #求出每一行第一个非0元素距离边框的位置（方式为从左往右计数和从右往左计数）
        first_nozeros, div_first_nozeros = self.get_first_nozeros(self.imagePA, Y_1, Y_2)
        #计算相关坐标
        pre_nozeros = first_nozeros[Y_1]
        div_pre_nozeros = div_first_nozeros[Y_1]
        for row in range(Y_1, Y_2):
            now_nozeros = first_nozeros[row]
            if now_nozeros == 0:
                now_nozeros += 1
            if float(pre_nozeros/now_nozeros) > 1.5:    #如果往下搜索过程中非0行所在列发生突变，认为找到肩部
                Y_ce = row
                X_eg = first_nozeros[row]
                break
            else:
                pre_nozeros = now_nozeros
        for row in range(Y_1, Y_2):
            div_now_nozeros = div_first_nozeros[row]
            if div_now_nozeros == 0:
                div_now_nozeros += 1
            if float(div_pre_nozeros/div_now_nozeros) > 1.5:
                Y_df = row
                X_fh =self.image_width-div_first_nozeros[row]-1
                break
            else:
                div_pre_nozeros = div_now_nozeros
        '''
        #计算cd两点所在列?????
        for col in range(X_eg+10, self.image_width):
            if self.imagePA[Y_ce][col] != 0:
                X_c = col
                break
        for col in range(self.image_width-X_fh+10, self.image_width-1):
            if self.imagePA[Y_df][self.image_width-col-1] != 0:
                X_d = self.image_width-col-1
                break
        '''
        #计算gh两点所在行
        for row in range(Y_ce, Y_2):
            if self.imagePA[row][X_eg] != 0:
                if self.imagePA[row+1][X_eg] !=0 and self.imagePA[row+2][X_eg] != 0:
                    Y_g = row

        for row in range(Y_df, Y_2):
            if self.imagePA[row][X_fh] != 0:
                if self.imagePA[row + 1][X_fh] != 0 and self.imagePA[row + 2][X_fh] != 0:
                    Y_h = row

        self.c = (self.a[0], Y_ce)
        self.d = (self.b[0], Y_df)
        self.e = (X_eg, Y_ce)
        self.f = (X_fh, Y_df)
        self.g = (X_eg, Y_g+10)
        self.h = (X_fh, Y_h+10)
        #重定位：
        '''
        X_5 = int(0.05 * self.image_width)
        X_50 = int(0.5 * self.image_width)
        X_95 = int(0.95 * self.image_width)
        x = 0
        min = 9999
        if X_eg <= X_5 or Y_ce < self.a[1]:
            Col_hist = np.sum(self.imagePA[:, X_5:X_50], axis=0) / self.imagePA.shape[0]
            # print(Col_hist)
            # print(len(Col_hist))
            for i in range(len(Col_hist)):
                if Col_hist[i] <= min:
                    min = Col_hist[i]
                    x = i
            X_eg = x + X_5
            print('e', X_eg)

        for Y in range(self.image_height):
            if self.imagePA[Y][X_eg] != 0:
                Y_ce = Y
                break
        if X_fh <= self.b[0] or Y_df < self.b[1]:
            Col_hist = np.sum(self.imagePA[:, X_50:X_95], axis=0) / self.imagePA.shape[0]
            # print(Col_hist)
            # print(len(Col_hist))
            for i in range(len(Col_hist)):
                if Col_hist[i] <= min:
                    min = Col_hist[i]
                    x = i
            X_fh = x + X_5
            print('f', X_fh)

        for Y in range(self.image_height):
            if self.imagePA[Y][X_fh] != 0:
                Y_df = Y
                break
        '''

        Col_hist = np.sum(self.imagePA, axis=0) / self.imagePA.shape[0]
        Middle_axis =int((self.a[0]+self.b[0])/2)

        #左肩正常右肩异常
        if Y_ce >= self.a[1]  and (Y_df < self.b[1] or X_fh <= self.b[0]):
            X_right = 2*Middle_axis - self.e[0]
            min = 255
            X_fh = 0
            range_right = np.minimum(X_right+20, self.image_width-1)



            for col in range(X_right-20, range_right):
                if Col_hist[col] < min and Col_hist[col] != 0:
                    min = Col_hist[col]
                    X_fh = col-10
            for row in range(self.b[1], self.b[1]+40):
                if self.imagePA[row][X_fh] != 0:
                    Y_df = row
                    break
            for row in range(Y_df, Y_2):
                if self.imagePA[row][X_fh] != 0:
                    if self.imagePA[row + 1][X_fh] != 0 and self.imagePA[row + 2][X_fh] != 0:
                        Y_h = row
            self.d = (self.b[0], Y_df)
            self.f = (X_fh, Y_df)
            self.h = (X_fh, Y_h+10)


        #左肩异常右肩正常：
        if Y_ce < self.a[1] and Y_df >= self.b[1]:
            X_left = 2*Middle_axis - self.f[0]
            min = 255
            X_eg = 0
            range_left = np.maximum(X_left-20, 0)
            for col in range(range_left, X_left+20):
                if Col_hist[col] < min and Col_hist[col]!= 0 :
                    min = Col_hist[col]
                    X_eg = col+10
            for row in range(self.a[1], self.a[1]+40):
                if self.imagePA[row][X_eg] != 0:
                    Y_ce = row
                    break
            for row in range(Y_ce, Y_2):
                if self.imagePA[row][X_eg] != 0:
                    if self.imagePA[row + 1][X_eg] != 0 and self.imagePA[row + 2][X_eg] != 0:
                        Y_g = row
            self.c = (self.a[0], Y_ce)
            self.e = (X_eg, Y_ce)
            self.g = (X_eg, Y_g+10)

        #肩高过小
        if self.g[1] - self.e[1] < 30:
            self.g = (self.e[0], self.e[1]+30)
        if self.h[1]-self.f[1] < 30:
            self.h = (self.f[0], self.f[1]+30)
    #获得图像中特定范围内的行的第一个非0元素离边框距离
    #param:
    #Y_1:搜索范围的起始行
    #Y_2:搜索范围的终止行
    #X_start:每一行从X_start列开始搜索直到列尾
    #X_div_start:每一行从右边第X_div_start列开始搜索直到列头
    #return 特定范围内的行的第一个非0元素离边框距离
    def get_first_nozeros(self, image, Y_1, Y_2, X_start=0, X_div_start=0):
        first_nozeros = []
        div_first_nozeros = []
        for row in range(image.shape[0]):
            first_nozeros.append(0)
            #first_nozeros[row] = imagePA.shape[1]-1 #搜索非0元素所在列前，将其初始化为最右边的点
            div_first_nozeros.append(0)
            #div_first_nozeros[row] = imagePA.shape[1]-1
        for row in range(Y_1, Y_2):
            for col in range(X_start, image.shape[1]):
                if image[row][col] != 0:
                    first_nozeros[row] = col
                    break
            for col in range(X_div_start, image.shape[1]):
                if image[row][image.shape[1]-col-1] != 0:
                    div_first_nozeros[row] = col
                    break
        return first_nozeros, div_first_nozeros
    def get_kl(self):
        Y_k = 0
        Y_l = 0
        X_k = 0
        X_l = 0
        first_nozeros, div_first_nozeros = self.get_first_nozeros(image=self.imagePA, Y_1=self.i[1], Y_2=self.i[1]+30, X_start=self.e[0], X_div_start=self.image_width-self.f[0]-1)
        Y_1 = self.i[1]    #注意Y_l与Y_1不是一个变量
        Y_2 = self.i[1]+30
        pre_nozeros = first_nozeros[Y_1]
        div_pre_nozeros = div_first_nozeros[Y_1]
        for row in range(Y_1, Y_2):
            now_nozeros = first_nozeros[row]
            if now_nozeros == 0:
                now_nozeros += 1
            if pre_nozeros - now_nozeros > 11:  # 如果往下搜索过程中非0行所在列发生突变，认为找到盆骨上端
                Y_k = row
                X_k = first_nozeros[row]
                break
            else:
                pre_nozeros = now_nozeros
        for row in range(Y_1, Y_2):
            div_now_nozeros = div_first_nozeros[row]
            if div_now_nozeros == 0:
                div_now_nozeros += 1
            if div_pre_nozeros - div_now_nozeros > 11:
                Y_l = row
                X_l = self.image_width - div_first_nozeros[row] - 1
                break
            else:
                div_pre_nozeros = div_now_nozeros

        self.k = (X_k, Y_k)
        self.l = (X_l, Y_l)
        # k l 点位置修正
        #k,l都出现问题：
        if X_l == X_k == Y_l == Y_k == 0:
            self.k = (self.i[0] - 15, min(self.i[1] + 10, self.image_height - 1))
            self.l = (min(self.j[0] + 15, self.image_width - 1), min(self.j[1] + 10, self.image_height - 1))
        #l正常，k出现问题：
        if (X_k == 0 or Y_k == 0) and X_l != 0 and Y_l != 0:
            self.k = (min(self.i[0]+self.j[0]-self.l[0], self.image_width-1), Y_1)
        #k正常，l出现问题：
        if (X_l == 0 or Y_l == 0) and X_k != 0 and Y_k != 0:
            self.l = (min(self.j[0]+self.i[0]-self.k[0], self.image_width-1), Y_k)



        #print("...............")
        #print(first_nozeros)
        #print(div_first_nozeros)
    def get_mn(self):
        Y_11 = int(self.image_height*0.11)
        Y_16 = int(self.image_height*0.16)
        Y_m = 0
        Y_n = 0

        list = []
        #从k点往下搜索全身的11%到16%处就行
        for row in range(self.k[1]+ Y_11, self.k[1]+Y_16):
            Flag = True
            for col in range(self.k[0], self.k[0]+10):
                if self.imagePA[row][col] != 0:
                    Flag = False
                    break
            if Flag == True:
                Y_m = row
                break

        for row in range(self.l[1]+Y_11, self.l[1]+Y_16):
            Flag = True
            for col in range(self.l[0]-10, self.l[0]):
                if self.imagePA[row][col] != 0:
                    Flag = False
                    break
            if Flag == True:
                Y_n = row
                break
        self.m = (self.k[0], Y_m)
        self.n = (self.l[0], Y_n)
        #重定位
        #左边正常右边异常
        if Y_m > self.k[1] and Y_n <= self.l[1]:
            Y_n = Y_m
            self.n = (self.l[0], Y_n)
        #右边正常左边异常
        if Y_m <= self.k[1] and Y_n > self.l[1]:
            Y_m = Y_n
            self.m = (self.k[0], Y_m)
        #两边都异常
        if Y_m <= self.k[1] and Y_n <= self.l[1]:
            Y_n = Y_m = self.k[1]+int(self.image_height*0.12)
            self.m = (self.k[0], Y_m)
            self.n = (self.l[0], Y_n)
        self.m = (self.k[0], int((Y_m+Y_n)/2))
        self.n = (self.l[0], int((Y_m + Y_n) / 2))
    def get_opst(self):
        first_nozeros, div_first_nozeros= self.get_first_nozeros(image=self.imageAP, Y_1=self.i[1], Y_2=self.i[1] + 50,
                                                                  X_div_start=self.e[0],
                                                                  X_start=self.image_width - self.f[0] - 1)
        Y_1 = self.i[1]
        Y_2 = self.i[1]+50
        min = self.image_width
        row_index = 0
        for row in range(Y_1, Y_2):
            if first_nozeros[row] < min:
                min = first_nozeros[row]
                row_index = row
        self.o = (min, row_index)


        min = self.image_width
        row_index = 0
        for row in range(Y_1, Y_2):
            if div_first_nozeros[row] < min:
                min =div_first_nozeros[row]
                row_index = row

        self.p = (self.image_width-min, row_index)
        self.s = (self.o[0]-10, self.o[1]-20)
        self.t = (self.p[0]+10, self.p[1]-20)
    def get_qr(self):
        first_nozeros, div_first_nozeros = self.get_first_nozeros(image=self.imagePA, Y_1=self.o[1], Y_2=self.image_height)

        return
    #得到手关节，脚关节，手掌，脚掌位置并返回结果。
    def get_joint(self):
        #左手关节位置:
        data = self.imagePA[:, 0:self.g[0]-5]
        exist = (data > 0) * 1.0
        left_hand = np.sum(exist, axis=1)
        max = 0
        left_hand_row = 0
        for row in range(self.i[1]-50, self.i[1]):
            if left_hand[row]+left_hand[row-1]+left_hand[row+1] > max:
                max = left_hand[row]+left_hand[row-1]+left_hand[row+1]
                left_hand_row = row
        #右手关节位置
        data = self.imagePA[:, self.h[0]+5:self.image_width-1]
        exist = (data > 0) * 1.0
        right_hand = np.sum(exist, axis=1)
        max = 0
        right_hand_row = 0
        for row in range(self.i[1] - 50, self.i[1]):
            if right_hand[row] + right_hand[row - 1] + right_hand[row + 1] > max:
                max = right_hand[row] + right_hand[row - 1] + right_hand[row + 1]
                right_hand_row = row


        #脚关节位置:
        exist = (self.imagePA > 0) * 1.0
        foot = np.sum(exist, axis=1)
        max = 0
        foot_joint_row = 0
        for row in range(self.m[1]+50, self.image_height-100):
            if foot[row]+foot[row-1]+foot[row+1] > max:
                max = foot[row]+foot[row-1]+foot[row+1]
                foot_joint_row = row
        if foot_joint_row == 0:
            foot_joint_row = int((self.m[1]+self.image_height-1)/2)


        #脚掌定位:
        max = 0
        foot_row = 0
        for row in range(self.image_height-100, self.image_height-1):
            if foot[row]+foot[row-1]+foot[row+1] > max:
                max = foot[row]+foot[row-1]+foot[row+1]
                foot_row = row
        if foot_row == 0:   #重定位
            foot_row = self.image_height-80

        self.foot_row = foot_row
        self.foot_joint_row = foot_joint_row
        self.left_hand_row = left_hand_row
        self.right_hand_row = right_hand_row
    #获取脊柱轮廓并修正
    def get_vertbra_contour(self):
        imagePA = self.imagePA.copy()
        cv.threshold(imagePA, 230, 255, cv.THRESH_TOZERO, imagePA)  # 简单阈值分割
        binary, contours, hierarchy = cv.findContours(imagePA, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) #脊柱轮廓粗提取
        max_area = 0
        max_contour = contours[0]
        for contour in contours:
            area = cv.contourArea(contour)
            if area > max_area:
                max_contour = contour
                max_area = area



        x, y, w, h = cv.boundingRect(max_contour)
        #轮廓信息转变为列表存储格式：
        max_contour.tolist()
        vertbra = []





        for points in max_contour:
            if points[0][1] >= self.a[1] and points[0][1] <= self.j[1]:
                vertbra.append((points[0][0], points[0][1]))
        '''      
        ImagePA = self.imagePA.copy()
        ImagePA= cv.cvtColor(ImagePA, cv.COLOR_GRAY2RGB)
        for point in vertbra:
            ImagePA[point[1], point[0]] = (0, 0, 255)



        cv.namedWindow("hello")
        cv.imshow("hell", ImagePA)
        cv.waitKey(0)
        '''

        #轮廓像素按升序排序
        vertbra_sort = vertbra.copy()




        def takeFirst(elem):
            return elem[0]
        vertbra_sort.sort(key=takeFirst)

        #线条闭合曲线拟合
        def get_connect(down, up, vertbra):
            length = len(vertbra)
            X_bias = vertbra[down][0] - vertbra[up][0]
            Y_bias = vertbra[down][1] - vertbra[up][1]
            list = []
            if X_bias < 0:
                for index in range(vertbra[down][0], vertbra[up][0]-1):
                    list.append((index+1, vertbra[down][1]))
                if Y_bias < 0:
                    for index in range(vertbra[down][1], vertbra[up][1]+1):
                        list.append((vertbra[up][0], index))
                if Y_bias > 0:
                    for index in range(vertbra[down][1], vertbra[up][1]-1, -1):
                        list.append((vertbra[up][0], index))
            if X_bias == 0:
                if Y_bias < 0:
                    for index in range(vertbra[down][1], vertbra[up][1]+1):
                        list.append((vertbra[up][0], index))
                if Y_bias > 0:
                    for index in range(vertbra[down][1], vertbra[up][1]-1, -1):
                        list.append((vertbra[up][0], index))
            if X_bias > 0:
                for index in range(vertbra[down][0], vertbra[up][0]+1, -1):
                    list.append((index-1, vertbra[down][1]))
                if Y_bias < 0:
                    for index in range(vertbra[down][1], vertbra[up][1]+1):
                        list.append((vertbra[up][0], index))
                if Y_bias > 0:
                    for index in range(vertbra[down][1], vertbra[up][1]-1, -1):
                        list.append((vertbra[up][0], index))

            if up < down:
                del vertbra[down + 1:length]
                vertbra.extend(list)
                del vertbra[0:up]
            if down < up:
                temp = vertbra[up+1:length]
                del vertbra[down+1:length]
                vertbra.extend(list)
                vertbra.extend(temp)

            return  vertbra
        #两点线段求所有点坐标:
        def get_points_list(pointA, pointB):
            list = []
            if pointA[0] == pointB[0]:
                for y in range(min(pointA[1], pointB[1]), max(pointA[1], pointB[1])+1):
                    list.append((pointA[0], y))
                return list
            if pointA[1] == pointB[1]:
                for x in range(min(pointA[0], pointB[0]), max(pointA[0], pointB[0])+1):
                    list.append((x, pointA[1]))
                return list

            #如果斜率大于1的话：
            if abs(pointA[1]-pointB[1]) >= abs(pointA[0]-pointB[0]):
                if pointA[1] < pointB[1]:
                    k = (pointA[0]-pointB[0])/(pointA[1]-pointB[1])  #斜率公式
                    for y in range(pointA[1], pointB[1]+1):
                        list.append((int((y-pointB[1])*k+pointB[0]), y))
                if pointA[1] > pointB[1]:
                    k = (pointA[0] - pointB[0]) / (pointA[1] - pointB[1])  # 斜率公式
                    for y in range(pointB[1], pointA[1] + 1):
                        list.append((int((y-pointB[1])*k+pointB[0]), y))

                return list



        #修正脊柱信息：
        #没找到脊柱（列表为空）:
        if len(vertbra_sort) == 0:
            list_line = get_points_list(self.a, self.i)
            list_line2 = get_points_list(self.b, self.j)
            list_line.extend(list_line2)
            vertbra = list_line.copy()
        else:
            # 左边像素越界太多
            while (self.a[0] + self.i[0]) / 2 - vertbra_sort[0][0] > 10:
                index = vertbra.index(vertbra_sort[0])
                up = down = index
                length = len(vertbra)
                Y_height = vertbra_sort[0][1]
                while vertbra[up][0] < (self.a[0] + self.i[0]) / 2 - 5 or (
                        vertbra[up][1] < Y_height + 20 and vertbra[up][1] > Y_height - 20):
                    up += 1
                    if up == length:
                        up = 0

                while vertbra[down][0] < (self.a[0] + self.i[0]) / 2 - 5 or (
                        vertbra[down][1] < Y_height + 20 and vertbra[down][1] > Y_height - 20):
                    down -= 1
                    if down == -1:
                        down = length - 1

                vertbra = get_connect(down, up, vertbra)

                Correction_information = self.Correction_information(vertbra)
                vertbra_sort = vertbra.copy()
                vertbra_sort.sort(key=takeFirst)
            # 右边像素越界太多
            while vertbra_sort[len(vertbra) - 1][0] - (self.b[0] + self.j[0]) / 2 > 10:
                index = vertbra.index(vertbra_sort[len(vertbra) - 1])
                up = down = index
                Y_height = vertbra_sort[len(vertbra) - 1][1]
                length = len(vertbra)
                while vertbra[up][0] - (self.b[0] + self.j[0]) / 2 - 5 > 0 or (
                        vertbra[up][1] < Y_height + 20 and vertbra[up][1] > Y_height - 20):
                    up += 1
                    if up == length:
                        up = 0
                while vertbra[down][0] - (self.b[0] + self.j[0]) / 2 - 5 > 0 or (
                        vertbra[down][1] < Y_height + 20 and vertbra[down][1] > Y_height - 20):
                    down -= 1
                    if down == -1:
                        down = length - 1
                vertbra = get_connect(down, up, vertbra)

                vertbra_sort = vertbra.copy()
                vertbra_sort.sort(key=takeFirst)


        #是否完全舍弃脊柱定位选择重置:


        if y > self.a[1] + 15:
            list_line = get_points_list(self.a, self.i)
            list_line2 = get_points_list(self.b, self.j)
            list_line.extend(list_line2)
            vertbra = list_line.copy()

        self.vertbra_contour = vertbra
    #列表按Y排序
    def Correction_information(self, vertbra):
        Correction_information = []  # 脊柱修正信息

        def takeSecond(elem):
            return elem[1]

        def takeFirst(elem):
            return elem[0]

        vertbra_sort = vertbra.copy()
        vertbra_sort.sort(key=takeSecond)
        temp_list = []
        # 信息处理1.排序显示
        for point in vertbra_sort:
            if len(temp_list):
                if temp_list[0][1] == point[1]:
                    temp_list.append(point)

                else:
                    temp_list.sort(key=takeFirst)
                    Correction_information.append(temp_list.copy())
                    temp_list.clear()
                    temp_list.append(point)
            else:
                temp_list.append(point)
        temp_list.sort(key=takeFirst)
        Correction_information.append(temp_list.copy())
        return Correction_information
    def get_vertbra_kidneys(self):
        image = self.OriginalPA.copy()
        vertbra_Y = self.Correction_information(self.vertbra_contour)
        black = (0, 0, 0)
        kidneys = []
        vertbra= []
        for list in vertbra_Y:

            for x in range(list[0][0], list[len(list) - 1][0]):

                image[list[0][1]][x] = 0
                vertbra.append((x, list[0][1]))

        # 将胸腔分成4个区域：

        image = median_blur_demo(image)


        area_1 = image[self.e[1]:int((self.e[1] + self.i[1]) / 2), self.e[0]:self.a[0]]
        area_2 = image[self.f[1]:int((self.f[1] + self.j[1]) / 2), self.b[0]:self.f[0]]
        area_3 = image[int((self.e[1] + self.i[1]) / 2):self.i[1], self.e[0]:self.a[0]]
        area_4 = image[int((self.f[1] + self.j[1]) / 2):self.j[1], self.b[0]:self.f[0]]
        regionGrow_3 = []
        regionGrow_4 = []



        if cv.mean(area_1)[0] < cv.mean(area_3)[0]*1.2:

            area_3_detect = area_3[int(area_3.shape[0]*0.5):int(area_3.shape[0]*0.8),int(area_3.shape[1]*0.5):int(area_3.shape[1])]

            _, max_3, _, max_index_3 = cv.minMaxLoc(area_3_detect)

            seeds_3 = [Point(max_index_3[1]+int(area_3.shape[0]*0.5),max_index_3[0]+int(area_3.shape[1]*0.5))]

            regionGrow_3 = regionGrow(area_3, seeds_3, max_3)

            for point in regionGrow_3:
                area_3[point[1], point[0]] = 0
                image[point[1]+int((self.e[1] + self.i[1]) / 2), point[0]+self.e[0]] = 0
                kidneys.append((point[0]+self.e[0], point[1]+int((self.e[1] + self.i[1]) / 2)))



            #cv.imshow("3", area_3)
            #cv.imshow("4", area_4)
        if cv.mean(area_2)[0] < cv.mean(area_4)[0]*1.2:
            area_4_detect = area_4[int(area_4.shape[0] * 0.5):int(area_4.shape[0] * 0.8), :int(area_4.shape[1] * 0.5)]
            _, max_4, _, max_index_4 = cv.minMaxLoc(area_4_detect)
            seeds_4 = [Point(max_index_4[1] + int(area_4.shape[0] * 0.5), max_index_4[0] + int(area_4.shape[1] * 0.1))]
            regionGrow_4 = regionGrow(area_4, seeds_4, max_4)
            for point in regionGrow_4:
                area_4[point[1], point[0]] = 0
                image[point[1]+int((self.f[1] + self.j[1]) / 2), point[0]+self.b[0]] = 0
                kidneys.append((point[0]+self.b[0], point[1]+int((self.f[1] + self.j[1]) / 2)))

        self.kidneys = kidneys
        self.vertbra = vertbra
    def get_points_list(self, pointA, pointB):
        list = []
        if pointA[0] == pointB[0]:
            for y in range(min(pointA[1], pointB[1]), max(pointA[1], pointB[1]) + 1):
                list.append((pointA[0], y))
            return list
        if pointA[1] == pointB[1]:
            for x in range(min(pointA[0], pointB[0]), max(pointA[0], pointB[0]) + 1):
                list.append((x, pointA[1]))
            return list

        # 如果斜率大于1的话：
        if abs(pointA[1] - pointB[1]) >= abs(pointA[0] - pointB[0]):
            if pointA[1] < pointB[1]:
                k = (pointA[0] - pointB[0]) / (pointA[1] - pointB[1])  # 斜率公式
                for y in range(pointA[1], pointB[1] + 1):
                    list.append((int((y - pointB[1]) * k + pointB[0]), y))
            if pointA[1] > pointB[1]:
                k = (pointA[0] - pointB[0]) / (pointA[1] - pointB[1])  # 斜率公式
                for y in range(pointB[1], pointA[1] + 1):
                    list.append((int((y - pointB[1]) * k + pointB[0]), y))
            return list

        # 如果斜率小于1的话：
        if abs(pointA[1] - pointB[1]) <= abs(pointA[0] - pointB[0]):
            k = (pointA[1] - pointB[1]) / (pointA[0] - pointB[0])  # 斜率公式
            if pointA[1] < pointB[1]:
                for y in range(pointA[1], pointB[1] + 1):
                    list.append((int((y - pointB[1]) * k + pointB[0]), y))
            if pointA[1] > pointB[1]:
                for y in range(pointB[1], pointA[1] + 1):
                    list.append((int((y - pointB[1]) * k + pointB[0]), y))
            return list

    def cut_head(self):
        imagehead1 = self.OriginalAP[0:self.a[1], self.image_width-self.head_right-1:self.image_width-self.head_left-1]
        imagehead2 = self.OriginalPA[0:self.a[1], self.head_left:self.head_right]
        return imagehead1, imagehead2
    def cut_shol(self):
        # self.get_cdefgh()
        image_pa = self.OriginalPA.copy()
        image_ap = self.OriginalAP.copy()
        c1 = (self.image_width - self.c[0], self.c[1])
        g1 = (self.image_width - self.g[0], self.g[1])
        d1 = (self.image_width - self.d[0], self.d[1])
        h1 = (self.image_width - self.h[0], self.h[1])
        list1 = self.get_points_list(self.c, self.g)
        list1.extend(self.get_points_list(self.c, (self.c[0], self.g[1])))
        list1 = self.Correction_information(list1)
        list2 = self.get_points_list(self.d, self.h)
        list2.extend(self.get_points_list(self.d, (self.d[0], self.h[1])))
        list2 = self.Correction_information(list2)
        list3 = self.get_points_list(c1, g1)
        list3.extend(self.get_points_list(c1, (self.image_width - self.c[0], self.g[1])))
        list3 = self.Correction_information(list3)
        list4 = self.get_points_list(d1, h1)
        list4.extend(self.get_points_list(d1, (self.image_width - self.d[0], self.h[1])))
        list4 = self.Correction_information(list4)
        for list in list1:
            for x in range(list[0][0], list[len(list) - 1][0] + 1):
                image_pa[list[0][1], x] = 0
        for list in list2:
            for x in range(list[0][0], list[len(list) - 1][0] + 1):
                image_pa[list[0][1], x] = 0
        for list in list3:
            for x in range(list[0][0], list[len(list) - 1][0]):
                image_ap[list[0][1], x] = 0
        for list in list4:
            for x in range(list[0][0], list[len(list) - 1][0]):
                image_ap[list[0][1], x] = 0

        imageshol_ap = image_ap[self.e[1]:self.g[1], self.image_width - self.c[0]:self.image_width]
        imageshol_pa = image_pa[self.e[1]:self.g[1], 0:self.c[0]]
        imageshor_ap = image_ap[self.f[1]:self.h[1], 0:self.image_width - self.d[0]]
        imageshor_pa = image_pa[self.f[1]:self.h[1], self.d[0]:self.image_width]
        return imageshol_ap, imageshol_pa, imageshor_ap, imageshor_pa
    def cut_pelvis(self):
        imagepelv_AP = self.OriginalAP[self.i[1]:self.m[1], self.o[0]:self.p[0]]
        imagepelv_PA = self.OriginalPA[self.j[1]:self.n[1], self.image_width-self.p[0]-1:self.image_width-self.o[0]-1]
        mean1, stddv1 = cv.meanStdDev(imagepelv_AP)
        mean2, stddv2 = cv.meanStdDev(imagepelv_PA)
        mean3, stddv3 = cv.meanStdDev(imagepelv_AP[int(imagepelv_AP.shape[0] * 0.5):int(imagepelv_AP.shape[0]),int(imagepelv_AP.shape[1] * 0.25):int(imagepelv_AP.shape[1] * 0.75)])
        mean4, stddv4 = cv.meanStdDev(imagepelv_PA[int(imagepelv_PA.shape[0]*0.5):int(imagepelv_PA.shape[0]), int(imagepelv_PA.shape[1]*0.25):int(imagepelv_PA.shape[1] * 0.75)])



        if stddv3 - stddv1 > 5:
            area_1 = imagepelv_AP[int(imagepelv_AP.shape[0] * 0.5):int(imagepelv_AP.shape[0]), int(imagepelv_AP.shape[1] * 0.25):int(imagepelv_AP.shape[1] * 0.75)]
            _, max_1, _, max_index_1 = cv.minMaxLoc(area_1)
            seeds_1 = [Point(max_index_1[1]+int(imagepelv_AP.shape[0]*0.5), max_index_1[0]+int(imagepelv_AP.shape[1]*0.25))]
            regionGrow_1 = regionGrow(imagepelv_AP, seeds_1, max_1)
            for point in regionGrow_1:
                imagepelv_AP[point[1], point[0]] = 0
        if stddv4 - stddv2 > 0:
            area_2 = imagepelv_PA[int(imagepelv_PA.shape[0]*0.5):int(imagepelv_PA.shape[0]), int(imagepelv_PA.shape[1]*0.25):int(imagepelv_PA.shape[1] * 0.75)]
            _, max_2, _, max_index_2 = cv.minMaxLoc(area_2)
            seeds_2 = [Point(max_index_2[1] + int(imagepelv_PA.shape[0] * 0.5), max_index_2[0] + int(imagepelv_PA.shape[1] * 0.25))]
            regionGrow_2 = regionGrow(imagepelv_PA, seeds_2, max_2)
            for point in regionGrow_2:
                imagepelv_PA[point[1], point[0]] = 0
                self.bladder.append((point[0] + self.image_width-self.p[0]-1, point[1] + self.j[1]))







        return imagepelv_AP, imagepelv_PA
        # 分割左右手关节部位
    def cut_elbow_joint(self):
        image_elbowl_ap = self.OriginalAP[self.left_hand_row - 20: self.left_hand_row + 20, self.image_width-1-self.g[0]:self.image_width-1]
        image_elbowl_pa = self.OriginalPA[self.left_hand_row - 20: self.left_hand_row + 20, 0:self.g[0]]
        image_elbowr_ap = self.OriginalAP[self.right_hand_row - 20: self.right_hand_row + 20, 0:self.image_width-self.h[0]-1]
        image_elbowr_pa = self.OriginalPA[self.right_hand_row - 20: self.right_hand_row + 20, self.h[0]:self.image_width-1]
        return image_elbowl_ap, image_elbowl_pa, image_elbowr_ap, image_elbowr_pa
        # 分割膝关节部位
    def cut_knee_joint(self):
        imageKneel_ap = self.OriginalAP[self.foot_joint_row - 25:self.foot_joint_row + 25,
                        self.image_width-1-int((self.e[0] + self.f[0]) / 2):self.image_width-1-self.e[0]]
        imageKneel_pa = self.OriginalPA[self.foot_joint_row - 25:self.foot_joint_row + 25,
                        self.e[0]:int((self.i[0] + self.j[0]) / 2)]
        imageKneer_ap = self.OriginalAP[self.foot_joint_row - 25:self.foot_joint_row + 25,
                        self.image_width-1-self.f[0]:self.image_width-1-int((self.e[0] + self.f[0]) / 2)]
        imageKneer_pa = self.OriginalPA[self.foot_joint_row - 25:self.foot_joint_row + 25,
                        int((self.i[0] + self.j[0]) / 2):self.f[0]]
        return imageKneel_ap, imageKneel_pa, imageKneer_ap, imageKneer_pa
    def cut_vertbra(self):
        image_AP = np.zeros(self.OriginalAP.shape, dtype=np.uint8)
        image_PA = np.zeros(self.OriginalPA.shape, dtype=np.uint8)
        for point in self.vertbra:
            image_AP[point[1], self.image_width - 1 - point[0]] = self.OriginalAP[point[1], self.image_width - 1 - point[0]]
            image_PA[point[1], point[0]] = self.OriginalPA[point[1], point[0]]
        minX = self.image_width
        maxX = 0
        minY = self.image_height
        maxY = 0

        for point in self.vertbra_contour:
            if point[0] < minX:
                minX = point[0]
            if point[0] > maxX:
                maxX = point[0]
            if point[1] < minY:
                minY = point[1]
            if point[1] > maxY:
                maxY = point[1]

        vertbra_ap = image_AP[minY:maxY, self.image_width-1-maxX:self.image_width-1-minX]
        vertbra_pa = image_PA[minY:maxY, minX:maxX]
        return vertbra_ap, vertbra_pa
    def cut_chest(self):
        image_AP = self.OriginalAP.copy()
        image_PA = self.OriginalPA.copy()
        #胸片切割前将脊柱和积水肾脏切除
        for point in self.vertbra:
            image_AP[point[1], self.image_width - 1 - point[0]] = 0
            image_PA[point[1], point[0]] = 0
        for point in self.kidneys:
            image_AP[point[1], self.image_width - 1 - point[0]] = 0
            image_PA[point[1], point[0]] = 0
        chestl_ap = image_AP[self.e[1]:self.i[1], self.image_width-1-int((self.a[0]+self.b[0])/2):self.image_width-1-self.e[0]]
        chestl_pa = image_PA[self.e[1]:self.i[1], self.e[0]:int((self.a[0]+self.b[0])/2)]
        chestr_ap = image_AP[self.e[1]:self.i[1], self.image_width-1-self.f[0]:self.image_width-1-int((self.a[0]+self.b[0])/2)]
        chestr_pa = image_PA[self.e[1]:self.i[1], int((self.a[0]+self.b[0])/2):self.f[0]]

        return chestl_ap, chestl_pa, chestr_ap, chestr_pa
    def cut_ankle_joint(self):
        image_AP = self.OriginalAP.copy()
        image_PA = self.OriginalPA.copy()
        ankle_jointl_ap = image_AP[self.foot_row-15:self.image_height-1, self.image_width-1-int((self.i[0]+self.j[0])/2):self.image_width-1-self.e[0]]
        ankle_jointl_pa = image_PA[self.foot_row-15:self.image_height-1, self.e[0]:int((self.i[0]+self.j[0])/2)]
        ankle_jointr_ap = image_AP[self.foot_row-15:self.image_height-1, self.image_width-1-self.f[0]:self.image_width-1-int((self.i[0]+self.j[0])/2)]
        ankle_jointr_pa = image_PA[self.foot_row-15:self.image_height-1, int((self.i[0]+self.j[0])/2):self.f[0]]

        return ankle_jointl_ap, ankle_jointl_pa, ankle_jointr_ap, ankle_jointr_pa

    def Region_Dectect(self, image_detect):
        def Region_points(x, y, height, width):
            if x > 0 and x < width-1 and y > 0 and y < height-1:
                return [(x, y-1),(x, y+1),(x-1, y),(x-1,y-1),(x-1,y+1),(x+1,y),(x+1,y-1),(x+1,y-1)]
            else:
                return [(x, y)]
        def takeFirst(elem):
            return elem[0]
        image = image_detect.copy()
        image = contrast_brightness_image(image, 50.0 / cv.mean(image)[0], 0)
        image_mask = np.zeros(image_detect.shape, dtype=np.uint8)
        height, width = image.shape
        count = 0
        mean = 0.0
        stddv = 0.0
        temp_list = []
        No_zero_list = []
        Region_max_points = []          #局部最大值
        for row in range(height):
            for col in range(width):
                if image[row, col] != 0:
                    No_zero_list.append(image[row, col])
        mean = np.mean(No_zero_list)
        stddv = np.std(No_zero_list)
        for row in range(height):
            for col in range(width):
                temp_list = Region_points(col, row, height, width)
                flag = True
                if image[row, col] <= mean*2.0:
                    continue
                for point in temp_list:
                    if image[row, col] <= image[point[1], point[0]]:
                        flag = False
                if flag:
                    Region_max_points.append((col, row))

        Region_max_points.sort(key=takeFirst)
        for point in Region_max_points:
            image_mask[point[1], point[0]] = 255
        #cv.imshow("", image_mask)

        #区域成长算法
        S_local = []
        Q_check = []
        for point in Region_max_points:  #Step 2
            S_local = []
            Q_check = []
            #S_local.append(point)       #Step 2
            Q_check.append(point)       #Step 2
            while len(Q_check):
                point_a = Q_check.pop()
                S_local.append(point_a)  # Step 3·
                temp_list.clear()
                temp_list = Region_points(point_a[0], point_a[1], height, width)
                Threshold = image[point_a[1], point_a[0]] * 0.9
                for point_b in temp_list:
                    if image[point_b[1], point_b[0]] >= Threshold and image_mask[point_b[1], point_b[0]] == 0:# and image[point_b[1], point_b[0]] <= image[point_a[1], point_a[0]]:
                        image_mask[point_b[1], point_b[0]] = 255
                        Q_check.append(point_b)
                        S_local.append(point_b)
            if len(S_local) < 13:
                for point in S_local:
                    image_mask[point[1], point[0]] = 0

        image_fuse = np.zeros((image.shape[0], image.shape[1]*3), dtype=np.uint8)
        image_fuse[:, :image.shape[1]] = image.copy()
        image_fuse[:, image.shape[1]:image.shape[1]*2] = image_mask.copy()
        for point in S_local:
            image[point[1], point[0]] = 255
        image_fuse[:, image.shape[1]*2:] = image.copy()

        cv.imshow("", image_fuse)
        cv.waitKey(10)
        return image
    #图片边缘修整
    def Cut_Region(self,*images):
        image_list = []
        #将多余部分剪除
        for image in images:
            rows = np.sum(image, axis=1)
            cols = np.sum(image, axis=0)
            minX, minY = 0, 0
            maxY, maxX = image.shape
            #边界确定
            for digit in rows:
                if digit == 0:
                    minY += 1
                else:
                    break
            while True:
                if maxY == 0:
                    break
                if rows[maxY-1] == 0:
                    maxY -= 1
                    if maxY == 1:
                        break
                else:
                    break
            for digit in cols:
                if digit == 0:
                    minX += 1
                else:
                    break
            while True:
                if maxX == 0:
                    break
                if cols[maxX-1] == 0:
                    maxX -= 1
                    if maxX == 1:
                        break
                else:
                    break
            temp = image[minY:maxY, minX:maxX]
            image_list.append(temp)
        return image_list
    #图片Size统一
    def Image_Resize(self,image_list):
        ap_row = [87, 34, 35, 153, 153, 39, 39, 168, 93, 49, 49, 61, 61]
        ap_col = [59, 59, 65, 56, 57, 35, 39, 28, 111, 60, 60, 61, 59]
        temp = []
        for i in range(13):
            image = cv.resize(image_list[i], (ap_col[i], ap_row[i]), interpolation=cv.INTER_AREA)
            temp.append(image)


        return temp




    def Region_Detect(self, image_detect):
        image = image_detect.copy()
        image_mask = np.zeros(image_detect.shape, dtype=np.uint8)
        image_mask_fix = np.zeros(image_detect.shape, dtype=np.uint8)
        map = np.zeros(image_detect.shape, dtype=np.uint8)          #3张掩码图片用途不一
        height, width = image.shape
        No_zero_list = []
        for row in range(height):
            for col in range(width):
                if image[row, col] != 0:
                    No_zero_list.append(image[row, col])
        image_mean = np.mean(No_zero_list)
        image_std = np.std(No_zero_list)
        #print(image_mean, image_std)
        #区域检测算法流程：
        #1.首先计算图像均值，保留像素值大于均值的元素
        #2.对保留下来的元素进行聚类检测，对于不聚类的元素删除
        #3.对于每一个聚类的集合用列表表示，整张图像的聚类集合用列表的列表表示。
        #4.基于SD进行筛选
        #1.
        for row in range(height):
            for col in range(width):
                if image[row, col] >image_mean+2.5*image_std or image[row, col] < image_mean-2.5*image_std:
                    image_mask[row, col] = 255



        # cv.imshow("nihaoa", image)
        # cv.imshow("hello", image_mask)

        #2.
        image_cluster_list = [] #整张图像的聚类表
        cluster_list = []       #单独一个聚类集合
        region = []
        def Region_points(x, y, height, width):
            if x > 0 and x < width-1 and y > 0 and y < height-1:
                return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            else:
                return []
        #:递归聚类算法：
        def test(point_list, image_mask):
            for point in point_list:
                if image_mask[point[1], point[0]] == 255 and map[point[1], point[0]] != 255:
                    cluster_list.append(point)
                    map[point[1], point[0]] = 255
                    region = Region_points(point[0], point[1], height, width)
                    test(region, image_mask)
        for y in range(height):
            for x in range(width):
                test([(x, y)], image_mask)
                #if len(cluster_list) > 13:
                    #print(cluster_list)
                if len(cluster_list) <= 13:
                    cluster_list.clear()
                    continue
                image_cluster_list.append(cluster_list.copy())
                cluster_list.clear()

        #print("hello")
        image_mask_fix = cv.cvtColor(image_mask_fix, cv.COLOR_GRAY2RGB)
        red = [0, 0, 255]
        green = [0, 255, 0]
        blue = [255, 0, 0]
        color = []
        color.append(blue)
        color.append(green)
        color.append(red)
        count = 0

        for cluster in image_cluster_list:
            count = (count + 1) % 3
            for point in cluster:
                image_mask_fix[point[1], point[0]] = color[count]

        cv.imshow("nide", image_mask_fix)


        return image_cluster_list
    #特征提取
    #输入：热点候选区域，被检测图像
    #输出：热点特征值
    def Feature_Detect(self, image_cluster_list, image_detect):
        feature_list = []
        for cluster_list in image_cluster_list:
            for point in cluster_list:
                image_detect[point[1], point[0]] = 255
            #cv.imshow("nihao",image_detect)
            # cv.waitKey(1000)
            #几何特征:面积、热点宽度、高度、周长、质心x、质心y、方向角、
            area = len(cluster_list)    #面积特征
            print("像素聚类：", cluster_list)
            temp = tuple(map(sorted, zip(*cluster_list)))
            min_x, max_x, min_y, max_y = temp[0][0], temp[0][-1], temp[1][0], temp[1][-1]
            height, width = max_y-min_y, max_x-min_x    #热点高度，宽度

            cluster_list_copy = np.array(cluster_list)
            mu = cv.moments(cluster_list_copy)          #计算列表的3阶及以下矩阵
            length = cv.arcLength(cluster_list_copy, True)  #热点周长
            print("周长：", length)
            print("面积：", area)
            print("3阶矩：", mu)
            if mu['m00'] != 0:
                point_x, point_y = mu['m10'] / mu['m00'], mu['m01'] / mu['m00']  # 质心
                # 根据二阶矩计算物体形状的方向，用theta表示
                r1 = mu['m20'] / mu['m00'] - point_x * point_x
                r2 = 2.0 * (mu['m11'] / mu['m00'] - point_x * point_y)
                r3 = mu['m02'] / mu['m00'] - point_y * point_y
                # print r1-r3
                if r1 - r3 == 0:
                    theta = np.pi / 2
                else:
                    theta = np.arctan(r2 / (r1 - r3)) / 2
                print("方向角：", theta)



            print("hello")


        return feature_list