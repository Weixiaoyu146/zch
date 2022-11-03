import os
import cv2
import json
import base64
import xml.etree.ElementTree as ET


def parse_img_label(img_path, xml_path):  # 绝对路径
    img = cv2.imread(img_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    bboxes = []  # 坐标框
    h, w = img.shape[0], img.shape[1]
    # gt_labels = []  # 标签名
    for obj in objs:  # 遍历所有的目标
        obj_is_deleted = obj[1].text
        if obj_is_deleted == '1':
            continue
        label = obj[0].text  # <name>这个tag的值，即标签
        label = label.strip(' ')
        x = set()
        y = set()
        for i in range(1, 5):
            x.add(int(obj[9][i][0].text))
            y.add(int(obj[9][i][1].text))
        xmin, ymin, xmax, ymax = min(x) - 1, min(y) - 1, max(x), max(y)
        assert (xmax > xmin)
        assert (ymax > ymin)
        box = [xmin, ymin, xmax, ymax]
        # box = [int(obj[9][i].text) for i in range(4)]
        box.append(label)  # box的元素 x1 y1 x2 y2 类别
        bboxes.append(box)
    return img, bboxes


# 该函数用于将yolo的标签转回xml需要的标签。。即将归一化后的坐标转为原始的像素坐标
def convert_yolo_xml(box, img):  #
    x, y, w, h = box[0], box[1], box[2], box[3]
    # 求出原始的x1 x2 y1 y2
    x2 = (2 * x + w) * img.shape[1] / 2
    x1 = x2 - w * img.shape[1]

    y2 = (2 * y + h) * img.shape[0] / 2
    y1 = y2 - h * img.shape[0]
    new_box = [x1, y1, x2, y2]
    new_box = list(map(int, new_box))
    return new_box


# 该函数用于解析yolo格式的数据集，即txt格式的标注 返回图像 边框坐标 真实标签名（不是索引，因此需要预先定义标签）
def parse_img_txt(img_path, txt_path):
    name_label = ['class0', 'class1', 'class2']  # 需要自己预先定义,它的顺序要和实际yolo格式的标签中0 1 2 3的标签对应 yolo标签的类别是索引 而不是名字
    img = cv2.imread(img_path)
    f = open(txt_path)
    bboxes = []
    for line in f.readlines():
        line = line.split(" ")
        if len(line) == 5:
            obj_label = name_label[int(line[0])]  # 将类别索引转成其名字
            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])
            box = convert_yolo_xml([x, y, w, h], img)
            box.append(obj_label)
            bboxes.append(box)
    return img, bboxes


# 制作labelme格式的标签
# 参数说明 img_name： 图像文件名称
# txt_name: 标签文件的绝对路径，注意是绝对路径
# prefix： 图像文件的上级目录名。即形如/home/xjzh/data/ 而img_name是其下的文件名，如00001.jpg
# prefix+img_name即为图像的绝对路径。不该路径能出现中文，否则cv2读取会有问题
#
def get_json(img_name, txt_name, label_color_dict, yolo=False):
    # 图片名 标签名 前缀
    label_dict = {}  # json字典，依次填充它的value
    label_dict["imagePath"] = img_name  # 图片路径
    label_dict["fillColor"] = [0, 0, 0, 128]  # 目标区域的填充颜色 RGBA
    label_dict["lineColor"] = [0, 255, 0, 128]  # 线条颜色
    label_dict["flag"] = {}
    label_dict["version"] = "4.5.7"  # 版本号，随便
    with open(img_name, "rb") as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        base64_str = str(base64_data, 'utf-8')
        label_dict["imageData"] = base64_str  # labelme的json文件存放了图像的base64编码。这样如果图像路径有问题仍然能够打开文件

    img, gt_box = parse_img_label(img_name, txt_name) if not yolo else parse_img_txt(img_name, txt_name)  # 读取真实数据

    label_dict["imageHeight"] = img.shape[0]  # 高度
    label_dict["imageWidth"] = img.shape[1]

    shape_list = []  # 存放标注信息的列表，它是 shapes这个键的值。里面是一个列表，每个元素又是一个字典，字典内容是该标注的类型 颜色 坐标点等等
    # label_dict["shapes"] = [] # 列表，每个元素是字典。
    # box的元素 x1 y1 x2 y2 类别
    for box in gt_box:
        shape_dict = {}  # 表示一个目标的字典
        shape_dict["shape_type"] = "rectangle"  # 因为xml或yolo格式标签是矩形框标注，因此是rectangle
        shape_dict["fill_color"] = None  # 该类型的填充颜色
        shape_dict["line_color"] = label_color_dict[box[-1]] if box[-1] in label_color_dict.keys() else label_color_dict["o"] # 线条颜色 可以设置，或者根据标签名自己预先设定labe_color_dict
        shape_dict["flags"] = {}
        shape_dict["label"] = box[-1]  # 标签名
        shape_dict["points"] = [[box[0], box[1]], [box[2], box[3]]]
        # 通常contours是长度为1的列表，如果有分块，可能就有多个  # [[x1,y1], [x2,y2]...]的列表
        shape_list.append(shape_dict)

    label_dict["shapes"] = shape_list  #
    return label_dict

if __name__ == '__main__':
    label_color_dict = {"n": [0, 255, 0, 128],
                        "u": [255, 255, 0, 128],
                        "b": [0, 0, 255, 128],
                        "bmp": [0, 0, 255, 128],
                        "bp": [0, 0, 255, 128],
                        "t": [255, 0, 0, 128],
                        "tmp": [255, 0, 0, 128],
                        "tp": [255, 0, 0, 128],
                        "a": [0, 255, 255, 128],
                        "s": [255, 0, 255, 128],
                        "o": [160, 32, 240, 128]}
    imgs_path = "../../Resources/PreprocessedImages/DenoisedImages/"  # 图像路径
    xmls_path = "../../Resources/Labels/Xml/"  # xml文件路径
    jsons_path = "../../Resources/Labels/Json/"  # 保存的json文件路径
    # jsons_path = "../../Resources/PreprocessedImages/DenoisedImages/"  # 保存的json文件路径

    xml_path = os.listdir(xmls_path)
    for nums, path in enumerate(xml_path):
        if nums % 200 == 0:
            print(f"processed {nums} xmls")
        img_path = imgs_path + path.replace('xml', 'jpg')  # img文件的绝对路径
        label_dict = get_json(img_path, xmls_path + path, label_color_dict)  #
        with open(jsons_path + path.replace("xml", "json"), 'w') as f:  # 写入一个json文件
            f.write(json.dumps(label_dict, ensure_ascii=False, indent=4, separators=(',', ':')))
