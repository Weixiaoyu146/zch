import os
import pydicom

def emptydircleansing(file_dir):
    for file_dir2 in os.listdir(file_dir):
        if os.listdir(file_dir + r'/' + file_dir2) == []:
            os.rmdir(file_dir + r'/' + file_dir2)


def errordicomcleansing(file_dir):
    for file_dir2 in os.listdir(file_dir):
        for file in os.listdir(file_dir + r'/' + file_dir2):
            if ".dcm" in file:
                try:
                    images = pydicom.dcmread(file_dir + r'/' + file_dir2 + r'/' + file)
                except Exception as e:
                    print(e.__class__.__name__, e)
                    os.remove(file_dir + r'/' + file_dir2 + r'/' + file)

            else:
                os.remove(file_dir + r'/' + file_dir2 + r'/' + file)

def errorsizecleansing(file_dir):
    # dcm_size_set = set()
    for file_dir2 in os.listdir(file_dir):
        for file in os.listdir(file_dir + r'/' + file_dir2):
            file_size = os.path.getsize(file_dir + r'/' + file_dir2 + r'/' + file)
            # dcm_size_set.add(file_size)
            if file_size < 1050000 or file_size > 1060000:
                os.remove(file_dir + r'/' + file_dir2 + r'/' + file)
                print(file_dir2)
    # dcm_size_list = list(dcm_size_set)
    # dcm_size_list.sort()
    # print

def stadicomname(file_dir):
    dicom_names_set = set()
    for file_dir2 in os.listdir(file_dir):
        for file in os.listdir(file_dir + r'/' + file_dir2):
            if ".dcm" in file:
                dicom_names_set.add(file)
    dicom_names_list = list(dicom_names_set)
    dicom_names_list.sort()
    for dicom_name in dicom_names_list:
        print(dicom_name)

def errornamecleansing(file_dir):
    for file_dir2 in os.listdir(file_dir):
        for file in os.listdir(file_dir + r'/' + file_dir2):
            if "ANT-" in file:
                os.remove(file_dir + r'/' + file_dir2 + r'/' + file)
            elif "ANTERIOR-" in file:
                os.remove(file_dir + r'/' + file_dir2 + r'/' + file)
            elif "POST-" in file:
                os.remove(file_dir + r'/' + file_dir2 + r'/' + file)
            elif "POSTERIOR-" in file:
                os.remove(file_dir + r'/' + file_dir2 + r'/' + file)


if __name__ == '__main__':
    file_dir = r'../../Resources/BoneScan'
    print("cleansing start")
    emptydircleansing(file_dir)
    # errordicomcleansing(file_dir)
    # emptydircleansing(file_dir)
    errorsizecleansing(file_dir)
    # emptydircleansing
    # stadicomname(file_dir)
    # errornamecleansing(file_dir)
    emptydircleansing(file_dir)
    print("cleansing end")
    # stadicomname(file_dir)