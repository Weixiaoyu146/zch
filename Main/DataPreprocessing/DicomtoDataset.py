import pymssql
import os
import pydicom
import shutil

def connect_sqlserver():
    try:
        conn = pymssql.connect(host='127.0.0.1:1433', user='sa', password='123', database='BoneScan', charset="UTF-8")
        cursor = conn.cursor()
    except Exception as e:
        print("连接数据库失败")
        print(e.__class__.__name__, e)
    return conn, cursor

def disconnect_sqlserver(conn, cursor):
    cursor.close()
    conn.close()

def get_id(cursor, sql_sentence):
    try:
        cursor.execute(sql_sentence)
        maxid = cursor.fetchone()
    except:
        print("查找失败")
    else:
        print("查找成功")
    finally:
        return maxid[0]

def get_image_path(file_dir):
    images_path = []
    for file_dir2 in os.listdir(file_dir):
        for file in os.listdir(file_dir + r'/' + file_dir2):
            if "ANT00" in file:
                images_path.append(file_dir + r'/' + file_dir2 + r'/' + file)
                break
            elif "Anterior" in file:
                images_path.append(file_dir + r'/' + file_dir2 + r'/' + file)
                break
            elif "ANTERIOR" in file:
                images_path.append(file_dir + r'/' + file_dir2 + r'/' + file)
                break
    return images_path

def add_sex_age(dicom_path, sql_sentence):
    try:
        dcm = pydicom.read_file(dicom_path)
    except:
        print('Dicom图打开失败')
        print(dicom_path)
    else:
        #性别
        if dcm.PatientSex == 'M':
            sql_sentence += "'" + '男' + "'" + ', '
        elif dcm.PatientSex == 'F':
            sql_sentence += "'" + '女' + "'" + ', '
        else:
            sql_sentence += 'NULL,'
        #年龄
        if dcm.PatientBirthDate != '':
            sql_sentence += str(int(dcm.StudyDate[0:4]) - int(dcm.PatientBirthDate[0:4]))  + ', '
        else:
            #缺省值为NULL
            sql_sentence += 'NULL, '

    return sql_sentence


if __name__ == '__main__':
    conn, cursor = connect_sqlserver()

    table_name = 'BoneScanImage'
    sql_sentence = 'select MAX(ID) from ' + table_name
    try:
        cursor.execute(sql_sentence)
        # conn.commit()
    except:
        print("查询最大ID失败")
    else:
        print("查询最大ID成功")
    finally:
        maxid = cursor.fetchone()[0]

    file_dir = r'../BoneScan'
    image_paths = get_image_path(file_dir)\

    sql_sentences = []
    for dicom_path in image_paths:
        sql_sentence = 'insert into ' + table_name + ' values(' + "'" + dicom_path[3:] + "'" + ', ' + str(maxid + 1) + ', '
        sql_sentence = add_sex_age(dicom_path, sql_sentence)
        sql_sentence += str(1) + ')'
        maxid += 1
        sql_sentences.append(sql_sentence)

    for sql_sentence in sql_sentences:
        try:
            cursor.execute(sql_sentence)
            conn.commit()
        except:
            print("插入失败")
            print(sql_sentence)
        finally:
            print("插入成功")
            print(sql_sentence)

    disconnect_sqlserver(conn, cursor)

    # a_files_name = ["ANTERIOR001_DS.dcm",
    #                 "ANTERIOR002_DS.dcm",
    #                 "ANTERIOR003_DS.dcm",
    #                 "ANTERIOR004_DS.dcm",
    #                 "ANTERIOR005_DS.dcm",
    #                 "ANTERIOR006_DS.dcm"]
    # p_files_name = ["POSTERIOR001_DS.dcm",
    #                 "POSTERIOR002_DS.dcm",
    #                 "POSTERIOR003_DS.dcm",
    #                 "POSTERIOR004_DS.dcm",
    #                 "POSTERIOR005_DS.dcm",
    #                 "POSTERIOR006_DS.dcm"]
    # for file_dir2 in os.listdir(file_dir):
    #     files = os.listdir(file_dir + r'/' + file_dir2)
        # if a_files_name[5] in files:
        #     if p_files_name[5] not in files:
        #         print("false")
        #         print(files)
        #         print(file_dir2)
                # shutil.rmtree(file_dir + r'/' + file_dir2)

        # if p_files_name[5] in files:
        #     if a_files_name[5] not in files:
        #         print("false")
        #         print(files)
        #         print(file_dir2)
        #         shutil.rmtree(file_dir + r'/' + file_dir2)

    # filename = set()
    # for file_dir2 in os.listdir(file_dir):
    #     for file in os.listdir(file_dir + r'/' + file_dir2):
    #         if ".dcm" in file:
    #             filename.add(file)
    #
    # filename = list(filename)
    # filename.sort()
    # for a in filename:
    #     print(a)



