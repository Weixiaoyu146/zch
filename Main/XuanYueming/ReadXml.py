import xml.etree.ElementTree as ET

labelPath = '../dataset/Annotations/'


def read_labels(xml_path):
    labels = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for object in root.iter('object'):
        x = set()
        y = set()
        for polygon in object.iter('polygon'):
            for pt in polygon.iter('pt'):
                x.add(int(pt[0].text))
                y.add(int(pt[1].text))
        label = [min(x), min(y), max(x), max(y)]
        label.append(object[0].text)
        labels.append(label)
    return labels


def match_2_labels(name, minX, minY, maxX, maxY):
    path = labelPath + name.split('.')[0] + '.xml'
    labels = read_labels(path)

    # 判断是否区域重叠
    for i in range(len(labels)):
        l = labels[i]
        if l[4] == 't' or l[4] == 'tp':
            # print(l[4])
            iw = min(maxX, l[2]) - max(minX, l[0])
            if iw > 0:
                ih = min(maxY, l[3]) - max(minY, l[1])
                if ih > 0:
                    return 1
    return 0


if __name__ == '__main__':

    xml_path = '../dataset\Annotations/2p.xml'
    print(read_labels(xml_path))
