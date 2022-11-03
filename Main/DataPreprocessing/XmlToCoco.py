import random
import sys
import os
import json
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {"N": 1,
                         "A": 2}
# PRE_DEFINE_CATEGORIES = {"Hotspot": 1}
# ANTERIOR_XML_DIR = 'AnteriorXmls/'
# POSTERIOR_XML_DIR = 'PosteriorXmls/'
# ANTERIOR_TRAIN_JSON = 'Anterior_train.json'
# ANTERIOR_TEST_JSON = 'Anterior_test.json'
# POSTERIOR_TRAIN_JSON = 'Posterior_train.json'
# POSTERIOR_TEST_JSON = 'Posterior_test.json'
XML_DIR = 'Xmls/'
IMAGE_DIR = '../../Resources/PreprocessedImages/GammaTransformedImages/'
TRAIN_JSON = 'train.json'
TEST_JSON = 'test.json'


# If necessary, pre-define category and its id


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename[:-1])
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def get_json_dict(json_dict, categories, bnd_id, xml_name):
    tree = ET.parse(xml_name)
    root = tree.getroot()
    path = get(root, 'path')
    if len(path) == 1:
        filename = os.path.basename(path[0].text)
    elif len(path) == 0:
        filename = get_and_check(root, 'filename', 1).text
    else:
        raise NotImplementedError('%d paths found in %s' % (len(path), xml_name))
    image_id = get_filename_as_int(filename)
    size = get_and_check(root, 'imagesize', 1)
    width = int(get_and_check(size, 'ncols', 1).text)
    height = int(get_and_check(size, 'nrows', 1).text)
    image = {'file_name': filename,
             'height': height,
             'width': width,
             'id': image_id}
    json_dict['images'].append(image)

    ## Cruuently we do not support segmentation
    #  segmented = get_and_check(root, 'segmented', 1).text
    #  assert segmented == '0'
    for obj in get(root, 'object'):
        obj_isdeleted = get_and_check(obj, 'deleted', 1).text
        if obj_isdeleted == '1':
            continue
        category = get_and_check(obj, 'name', 1).text
        if category == 'n':
            category = 'Normal'
            category = 'Hotspot'
        elif category == 'u':
            category = 'Uncertain'
            category = 'Hotspot'
        elif category == 'b' or category == 'bmp' or category == 'bp':
            category = 'Benign'
            category = 'Hotspot'
        elif category == 't' or category == 'tmp' or category == 'tp':
            category = 'Malignant'
            category = 'Hotspot'
        else:
            continue

        category_id = categories[category]
        x = set()
        y = set()
        bndbox = get_and_check(obj, 'polygon', 1)
        for point in get_and_check(bndbox, 'pt', 4):
            x.add(int(get_and_check(point, 'x', 1).text))
            y.add(int(get_and_check(point, 'y', 1).text))
        xmin, ymin, xmax, ymax = min(x) - 1, min(y) - 1, max(x), max(y)
        assert (xmax > xmin)
        assert (ymax > ymin)
        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
            image_id, 'bbox': [xmin, ymin, o_width, o_height],
               'category_id': category_id, 'id': bnd_id, 'ignore': 0}
        json_dict['annotations'].append(ann)
        bnd_id = bnd_id + 1
    return json_dict, bnd_id


def convert(txt_path, root_path, wl_ww):
    xml_list = open(txt_path, 'r')
    json_dict = {"images": [],
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID

    for line in xml_list:
        line = line.strip()
        print("Processing %s" % (line))
        # ant_image_path = IMAGE_DIR + line + 'a.jpg'
        # pos_image_path = IMAGE_DIR + line + 'p.jpg'
        # ant_image = cv2.imread(ant_image_path, cv2.IMREAD_GRAYSCALE)
        # pos_image = cv2.imread(pos_image_path, cv2.IMREAD_GRAYSCALE)
        # images = np.hstack((ant_image, pos_image))
        # images_path = os.path.join(root_path, r'OriginalImages/' + line + '.jpg')
        # cv2.imwrite(images_path, images, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        ant_xml_name = os.path.join(root_path, XML_DIR + line + 'a.xml')
        pos_xml_name = os.path.join(root_path, XML_DIR + line + 'p.xml')
        file_name = line + '.jpg'
        image = {'file_name': file_name,
                 'height': 1024,
                 'width': 512,
                 'id': int(line)}
        json_dict['images'].append(image)

        if os.path.exists(ant_xml_name):
            tree = ET.parse(ant_xml_name)
            root = tree.getroot()
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                # if category == 'Normal' or category == 'Artifactual' or category == 'Malignant' or category == 'Benign' or category == 'Uncertain':
                #     category = 'H'
                if category == 'Normal' or category == 'Artifactual':
                    category = 'N'
                elif category == 'Uncertain' or category == 'Benign' or category == 'Malignant':
                    category = 'A'
                # elif category == 'Uncertain':
                #     category = 'U'
                # elif category == 'Benign':
                #     category = 'B'
                # elif category == 'Malignant':
                #     category = 'M'
                else:
                    continue

                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bndbox, 'xmin', 1).text)
                ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                if category == 'U':
                    ann1 = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': int(line),
                            'bbox': [xmin, ymin, o_width, o_height], 'category_id': categories['B'],
                            'id': bnd_id, 'ignore': 0}
                    ann2 = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': int(line),
                            'bbox': [xmin, ymin, o_width, o_height], 'category_id': categories['M'],
                            'id': bnd_id + 1, 'ignore': 0}
                    json_dict['annotations'].append(ann1)
                    json_dict['annotations'].append(ann2)
                    bnd_id += 2
                else:
                    ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': int(line),
                           'bbox': [xmin, ymin, o_width, o_height], 'category_id': categories[category],
                           'id': bnd_id, 'ignore': 0}
                    json_dict['annotations'].append(ann)
                    bnd_id += 1

        if os.path.exists(pos_xml_name):
            tree = ET.parse(pos_xml_name)
            root = tree.getroot()
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                # if category == 'Normal' or category == 'Artifactual' or category == 'Malignant' or category == 'Benign' or category == 'Uncertain':
                #     category = 'H'
                if category == 'Normal' or category == 'Artifactual':
                    category = 'N'
                elif category == 'Uncertain' or category == 'Benign' or category == 'Malignant':
                    category = 'A'
                # elif category == 'Uncertain':
                #     category = 'U'
                # elif category == 'Benign':
                #     category = 'B'
                # elif category == 'Malignant':
                #     category = 'M'
                else:
                    continue

                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bndbox, 'xmin', 1).text) + 256
                ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                xmax = int(get_and_check(bndbox, 'xmax', 1).text) + 256
                ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                if category == 'U':
                    ann1 = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': int(line),
                            'bbox': [xmin, ymin, o_width, o_height], 'category_id': categories['B'],
                            'id': bnd_id, 'ignore': 0}
                    ann2 = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': int(line),
                            'bbox': [xmin, ymin, o_width, o_height], 'category_id': categories['M'],
                            'id': bnd_id + 1, 'ignore': 0}
                    json_dict['annotations'].append(ann1)
                    json_dict['annotations'].append(ann2)
                    bnd_id += 2
                else:
                    ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': int(line),
                           'bbox': [xmin, ymin, o_width, o_height], 'category_id': categories[category],
                           'id': bnd_id, 'ignore': 0}
                    json_dict['annotations'].append(ann)
                    bnd_id += 1
        # json_dict, bnd_id = get_json_dict(json_dict, categories, bnd_id, ant_xml_name)

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    if 'train' in txt_path:
        json_file = os.path.join(root_path, r'Annotations/' + wl_ww + TRAIN_JSON)
    elif 'test' in txt_path:
        json_file = os.path.join(root_path, r'Annotations/' + wl_ww + TEST_JSON)
    # if 'ant_train' in txt_path:
    #     json_file = os.path.join(root_path, r'Annotations/' + wl_ww + ANTERIOR_TRAIN_JSON)
    # elif 'ant_test' in txt_path:
    #     json_file = os.path.join(root_path, r'Annotations/' + wl_ww + ANTERIOR_TEST_JSON)
    # elif 'pos_train' in txt_path:
    #     json_file = os.path.join(root_path, r'Annotations/' + wl_ww + POSTERIOR_TRAIN_JSON)
    # elif 'pos_test' in txt_path:
    #     json_file = os.path.join(root_path, r'Annotations/' + wl_ww + POSTERIOR_TEST_JSON)
    #
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

    xml_list.close()
    print('end')


def train_test_spilt(xml_dir, txt_dir, train_rate):
    # xml_list = get_abnormal(xml_dir)
    xml_list = get_xml_list(xml_dir)
    xml_num = len(xml_list)
    list = range(xml_num)
    train_num = int(train_rate * xml_num)
    train = random.sample(list, train_num)

    # if 'Anterior' in xml_dir:
    #     train_path = '/ant_train.txt'
    #     test_path = '/ant_test.txt'
    # elif 'Posterior' in xml_dir:
    #     train_path = '/pos_train.txt'
    #     test_path = '/pos_test.txt'

    train_path = '/train.txt'
    test_path = '/test.txt'

    txt_train = open(txt_dir + train_path, 'w')
    txt_test = open(txt_dir + test_path, 'w')

    for i in list:
        name = xml_list[i] + '\n'
        if i in train:
            txt_train.write(name)
        else:
            txt_test.write(name)

    txt_train.close()
    txt_test.close()


def move_train_test(root_path, test_path, wl_ww):
    test_list = open(test_path, 'r')
    for line in test_list:
        line = line.strip()
        src_ant_image_dir = os.path.join(root_path, r'Posterior-23.5-47/' + line + 'p.jpg')
        dst_ant_image_dir = os.path.join(root_path, wl_ww + r'Posterior_test/' + line + 'p.jpg')
        if not os.path.exists(os.path.join(root_path, wl_ww + r'Posterior_test/')):
            os.mkdir(os.path.join(root_path, wl_ww + r'Posterior_test/'))
        shutil.move(src_ant_image_dir, dst_ant_image_dir)


def get_abnormal(xml_dir):
    xml_list = os.listdir(xml_dir)
    abnormal_list = []
    for xml_name in xml_list:
        abnormal_flag = False
        xml_path = os.path.join(xml_dir, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in get(root, 'object'):
            obj_isdeleted = get_and_check(obj, 'deleted', 1).text
            if obj_isdeleted == '1':
                continue
            category = get_and_check(obj, 'name', 1).text
            if category == 'u':
                abnormal_flag = True
            elif category == 'b' or category == 'bmp' or category == 'bp':
                abnormal_flag = True
            elif category == 't' or category == 'tmp' or category == 'tp':
                abnormal_flag = True
            else:
                continue
        if abnormal_flag == True:
            abnormal_list.append(xml_name)
    return abnormal_list


def get_xml_list(xml_dir):
    all_xml_list = os.listdir(xml_dir)
    xml_set = set()
    for xml_name in all_xml_list:
        xml_set.add(xml_name[:-5])
    xml_list = list(xml_set)
    return xml_list


def convertToRGB(root_path):
    all_image_list = os.listdir(IMAGE_DIR)
    image_set = set()
    for image_name in all_image_list:
        image_set.add(image_name[:-5])
    image_list = list(image_set)
    for image_name in image_list:
        print("Processing %s" % (image_name))
        ant_image_path = IMAGE_DIR + image_name + 'a.jpg'
        pos_image_path = IMAGE_DIR + image_name + 'p.jpg'
        ant_image = cv2.imread(ant_image_path, cv2.IMREAD_GRAYSCALE)
        pos_image = cv2.imread(pos_image_path, cv2.IMREAD_GRAYSCALE)
        images = np.hstack((ant_image, pos_image))
        # images_path = os.path.join(root_path, r'GammaTransformedImages/' + image_name + '.jpg')
        # images_hflip = cv2.flip(images, 1)
        # images_RGB = np.array([images, images_hflip, images])
        # images_RGB = np.moveaxis(images_RGB, 0, -1)
        # cv2.imwrite(images_path, images_RGB, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        images_path = os.path.join(root_path, r'GammaTransformedImages/' + image_name + '.jpg')
        cv2.imwrite(images_path, images, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def clearEmptyObjectXml(xml_dir):
    xml_list = os.listdir(xml_dir)
    emptyXmls = []
    for xml_name in xml_list:
        path = os.path.join(xml_dir, xml_name)
        tree = ET.parse(path)
        root = tree.getroot()
        objects = get(root, 'object')
        if not len(objects):
            emptyXmls.append(path)
    print(len(emptyXmls))
    print(emptyXmls)
    for emptyXml in emptyXmls:
        os.remove(emptyXml)
        if not os.path.exists(emptyXml):
            print(1)


def train_test(xml_dir, txt_dir):
    xml_list = get_xml_list(xml_dir)
    xml_num = len(xml_list)
    print(xml_num)

    test_path = '/test.txt'
    txt_test = open(txt_dir + test_path, 'r')
    lines = txt_test.readlines()
    test = []
    for line in lines:
        line = line.strip('\n')
        if line in test:
            print(line)
        test.append(line)
    txt_test.close()

    train_path = '/train.txt'
    exclude_path = '/exclude.txt'

    txt_train = open(txt_dir + train_path, 'w')
    txt_exclude = open(txt_dir + exclude_path, 'w')


    for i in range(1, 3254):
        if str(i) in xml_list:
            if str(i) not in test:
                txt_train.writelines(str(i))
                txt_train.write('\n')
        else:
            txt_exclude.writelines(str(i))
            txt_exclude.write('\n')

    txt_train.close()
    txt_exclude.close()
    txt_test.close()


def test_duplicate(txt_dir):
    train_path = '/exclude.txt'
    test_path = '/test.txt'
    txt_train = open(txt_dir + train_path, 'r')
    txt_test = open(txt_dir + test_path, 'r')
    train_lines = txt_train.readlines()
    test_lines = txt_test.readlines()
    for i in test_lines:
        if i in train_lines:
            print(i)


if __name__ == '__main__':
    # clearEmptyObjectXml(r'../../Resources/Dataset/Xmls')

    # train_test_spilt(r'../../Resources/Dataset/Xmls',
    #                  r'../../Resources/Dataset/Txts', 0.8)

    # train_test(r'../../Resources/Dataset/Xmls',
    #            r'../../Resources/Dataset/Txts')

    # test_duplicate(r'../../Resources/Dataset/Txts')

    convert(r'../../Resources/Dataset/Txts/train.txt',
            r'../../Resources/Dataset', 'NA-')

    convert(r'../../Resources/Dataset/Txts/test.txt',
            r'../../Resources/Dataset', 'NA-')

    # move_train_test(r'../../Resources/PreprocessedImages/Dataset',
    #                 r'../../Resources/PreprocessedImages/Dataset/Txts/test.txt', '23.5-47-')

    # get_abnormal(r'../../Resources/PreprocessedImages/Dataset/AnteriorXmls')

    # convertToRGB(r'../../Resources/PreprocessedImages/Dataset')
