import cv2.cv2
import numpy as np
import pydicom

if __name__ == '__main__':
    dicom_path = r"D:/Desktop/Test/Dicoms/NM3592839-Li XiuZhen_1.DCM"
    dicom = pydicom.dcmread(dicom_path)
    binary_image = dicom.pixel_array
    image_a = np.uint8(binary_image[0])
    image_p = np.uint8(binary_image[1])

    patient_name = dicom_path.split('.')[-2].split('-')[-1]
    print(dicom.PatientID)
    print(patient_name)
    print(image_a.shape)
    print(image_p.shape)
    cv2.imshow('1', image_a)
    cv2.imshow('2', image_p)
    cv2.waitKey(0)