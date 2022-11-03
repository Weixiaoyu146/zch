import os
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


def update_xmls(folder_path):
    exclude = ["2007", "2011", "2015", "2016", "2018", "2025", "2032", "2033", "2039", "2038", "2048", "2054"]
    xmls_list = os.listdir(folder_path)
    id = 1
    for xml in xmls_list:
        if xml[:-5] in exclude:
            continue
        xml_path = os.path.join(folder_path, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root:
            if "path" == child.tag:
                root.remove(child)
                break
        for child in root:
            if "folder" == child.tag:
                child.text = "BS-80K"
            elif "filename" == child.tag:
                child.text = str(id) + child.text[-4:]
            elif "source" == child.tag:
                for source_child in child:
                    if "database" == source_child.tag:
                        source_child.text = "The BS-80K Database"
            elif "object" == child.tag:
                for object_child in child:
                    if "name" == object_child.tag:
                        if "Normal" == object_child.text or "Artifactual" == object_child.text:
                            object_child.text = "Normal"
                        elif "Benign" == object_child.text or "Uncertain" == object_child.text or "Malignant" == object_child.text:
                            object_child.text = "Abnormal"
                    elif "difficult" == object_child.tag:
                        object_child.text = "1"
        if xml[-5:-4] == 'a':
            new_path = os.path.join(folder_path[:-7], "ant/" + str(id) + xml[-4:])
        elif xml[-5:-4] == 'p':
            new_path = os.path.join(folder_path[:-7], "post/" + str(id) + xml[-4:])
        tree.write(new_path)
        id += 1
    return 0


if __name__ == '__main__':
    folder_path = "../../Resources/Test/BS-80K-source"
    update_xmls(folder_path)

# xml_path = r"../Labels/20a.xml"
# print(read_labels(xml_path))
