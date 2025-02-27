#!/usr/bin/env python

import os, sys, wget, zipfile
from pathlib import Path

cadcd = {
    '2019_02_27': [
        '0031'
    ]
}
"""
cadcd = {
    '2018_03_06': [
        '0001','0002','0005','0006','0008','0009','0010',
        '0012','0013','0015','0016','0018'
    ],
    '2018_03_07': [
        '0001','0002','0004','0005','0006','0007'
    ],
    '2019_02_27': [
        '0002','0003','0004','0005','0006','0008','0009','0010',
        '0011','0013','0015','0016','0018','0019','0020',
        '0022','0024','0025','0027','0028','0030',
        '0031','0033','0034','0035','0037','0039','0040',
        '0041','0043','0044','0045','0046','0047','0049','0050',
        '0051','0054','0055','0056','0058','0059',
        '0060','0061','0063','0065','0066','0068','0070',
        '0072','0073','0075','0076','0078','0079',
        '0080','0082'
    ]
}
"""

dataset_path=''

if len(sys.argv) == 2:
    dataset_path = sys.argv[1]
    os.chdir(dataset_path)
else:
    print('Please enter the path to store the dataset.')
    exit()

labeled = True
if labeled:
    print('Downloading labeled data')
else:
    print('Downloading raw data')

def main():
    root_dir = Path(dataset_path + '/cadcd')
    root_dir.mkdir(parents=True, exist_ok=True)
    for date in cadcd:
        print(date)
        date_path = dataset_path + '/cadcd/' + date
        date_dir = Path(date_path)
        date_dir.mkdir(parents=True, exist_ok=True)

        # Download calibration for this date
        os.chdir(date_path)
        calib_url = 'http://wiselab.uwaterloo.ca/cadcd_data/' + date + '/calib.zip'
        calib_filename = wget.download(calib_url)
        # Extract files
        zip = zipfile.ZipFile(calib_filename)
        zip.extractall()
        zip.close()
        # Delete zip file
        os.remove(calib_filename)

        for drive in cadcd[date]:
            print(drive)
            drive_path = dataset_path + '/cadcd/' + date + '/' + drive
            drive_dir = Path(drive_path)
            drive_dir.mkdir(parents=True, exist_ok=True)

            # Download drive
            os.chdir(drive_path)
            base_url = 'http://wiselab.uwaterloo.ca/cadcd_data/' + date + '/' + drive
            if labeled:
                data_url = base_url + '/labeled.zip'
                ann_3d_url = base_url + '/3d_ann.json'
                ann_3d_filename = wget.download(ann_3d_url)
            else:
                data_url = base_url + '/raw.zip'
            data_filename = wget.download(data_url)

            # Extract files
            zip = zipfile.ZipFile(data_filename)
            zip.extractall()
            zip.close()

            # Delete zip file
            os.remove(data_filename)

main()
