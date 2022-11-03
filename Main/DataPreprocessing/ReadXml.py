import xml.etree.ElementTree as ET

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

xml_path = r"../Labels/20a.xml"
print(read_labels(xml_path))
