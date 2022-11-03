import cv2.cv2

from Preprocessing import *
from Segmentation import Segmentation
from contrast_brightness import ContrastAugmentation


def test_preprocessing():
    # 　加载所有图像
    images, nums = load_pairs('./images_test/')
    # print(np.shape(images))

    # 　原图像分割
    for i in range(nums):
        resultA = get_bone(images[i][0])
        resultP = get_bone(images[i][1])
        originA = image_enhance(images[i][0])
        originP = image_enhance(images[i][1])
        segmentation = Segmentation(resultA, resultP, originA, originP,
                                    str(i + 1) + 'a.jpg', str(i + 1) + 'p.jpg')
        segmentation.Segment_pelvis()

# 图像分割
def segment(images):
    # fo = open("pelvis_labels.txt", "w")

    for pairs in images[:5]:
        resultA = get_bone(pairs[0]) # 提取骨架
        resultP = get_bone(pairs[1])
        segmentation = Segmentation(resultA, resultP, pairs[0], pairs[1],
                                    str(pairs[2]) + 'A.jpg', str(pairs[2]) + 'P.jpg')
        # segmentation.show_point()
        # try:
        segmentation.Get_points()  # 得到分割部位，并增强后保存
        segmentation.Segment_vertbra()
        # except:
        #     print(str(pairs[2]))
        #     continue
        # cv.imwrite('./seg_test/vertbra/' + segmentation.nameP,
        #            segmentation.show_segment(), [int(cv.IMWRITE_JPEG_QUALITY), 95])

        # fo.write(str(pairs[2]) + ' ---> A: ' + str(segmentation.labels_pelvis[0]) \
        #          + ', P: ' + str(segmentation.labels_pelvis[1]) + '\n')

        # print(str(pairs[2]) + ' ---> A: ' + str(segmentation.labels_pelvis[0]) \
        #           + ', P: ' + str(segmentation.labels_pelvis[1]))

        # 绘制某图像的定位点和框
        # if pairs[2] in [0, 710]:
        segmentation.show_point()
        # segmentation.show_segment()
        # cv.imshow('segV', segmentation.show_segment())

    # fo.close()

    return segmentation


if __name__ == '__main__':
    # 加载所有图像：前身图，后身图，id
    images = load_images(r'../../Resources/PreprocessedImages/DenoisedImages')
    print('病例个数：' + str(len(images)))

    for image in images:
        cv2.cv2.imwrite(r'../../Resources/PreprocessedImages/AugmentedImages/XuanYueming/' + str(image[2]) + 'a.jpg',
                        image_enhance(image[0], image[2]), [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv2.cv2.imwrite(r'../../Resources/PreprocessedImages/AugmentedImages/XuanYueming/' + str(image[2]) + 'p.jpg',
                        image_enhance(image[1], image[2]), [int(cv.IMWRITE_JPEG_QUALITY), 100])
        # image_enhance(image[0], image[2])
        # image_enhance(image[1], image[2])

        # image_enhance(image[0], image[2])
        # ContrastAugmentation(image[0], image[2], 23.5, 47)
        cv2.cv2.imwrite(r'../../Resources/PreprocessedImages/AugmentedImages/PingMing/' + str(image[2]) + 'a.jpg',
                        ContrastAugmentation(image[0], image[2], 23.5, 47), [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv2.cv2.imwrite(r'../../Resources/PreprocessedImages/AugmentedImages/PingMing/' + str(image[2]) + 'p.jpg',
                        ContrastAugmentation(image[1], image[2], 23.5, 47), [int(cv.IMWRITE_JPEG_QUALITY), 100])


    # image_enhance(images[1][1])
    # get_bone(images[0][1])

    # segmentation = segment(images)

    cv.waitKey(0)
    print("---------------The end------------------")
