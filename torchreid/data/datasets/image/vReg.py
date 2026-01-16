# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os
import sys
import os.path as osp
import re
from lxml import etree
from torchreid.utils import read_image
from ..dataset import ImageDataset
import numpy as np
import matplotlib.pyplot as plt
import json
class vReg(ImageDataset):
    """vReg.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "vReg"
    dataset_name = "vReg"

    train_xml_path = "train_label.xml"
    test_xml_path = "test_label.xml"


    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.use_xml_file = False # json 형식 파일을 읽을 때 False로 둔다. 2025.04.48 by 윤경섭
        if self.use_xml_file:
            self.train_xml_file = osp.join(self.dataset_dir, self.train_xml_path)
            self.test_xml_file = osp.join(self.dataset_dir, self.test_xml_path)
        else:
            self.train_xml_file = None
            self.test_xml_file = None
        self.query_xml_file = None
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir,self.train_xml_file)
        query = self.process_dir(self.query_dir, self.query_xml_file, is_train=False)
        gallery = self.process_dir(self.gallery_dir, self.test_xml_file, is_train=False)
        
        super(vReg, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        """
        arr = np.array(img)
        arr_hwc = np.transpose(arr, (1, 2, 0)) 
        # 3) 그리기
        plt.figure(figsize=(4, 8))
        plt.imshow(arr_hwc)                  # arr이 (H, W, 3)이므로 그대로 사용
        plt.title(img_path.split('/')[-1])
        plt.axis('off')
        plt.show()
        """
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item
    
    @staticmethod
    def load_map(file_path):
        mapping = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    name, num = line.split()
                    mapping[name.strip()] = int(num.strip())
        return mapping    

    def process_dir(self, dir_path, xml_file, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        if xml_file:
            pattern = re.compile(r'([\d]+)_c(\d\d\d)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')         #market1501 type

        #주의 vReg 의 데이터셋에서 주의 할 점은 파일의 pid 갯수와 xml 파일의 pid 갯수가 다르다는 것이다.
        # 예를 들어 파일의 pid 갯수는 576개이고 xml 파일의 pid 갯수는575개이다.
        pid_container = set()

        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        #여기서 color_id와 type_id의 범위는 무조건 정의 되어 있는 갯수라고 생각하면 된다.
        if xml_file:
            # lxml에서는 XMLParser에 다중 바이트 인코딩을 지정 가능
            parser = etree.XMLParser(encoding="gb2312")
            tree = etree.parse(xml_file, parser)
            root = tree.getroot()

            # 'Items' 엘리먼트를 찾습니다.
            items = root.find('Items')

            for item in items.findall('Item'):
                pid = int(item.attrib.get('vehicleID'))
                if pid == -1:
                    continue # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}


            for item in items.findall('Item'):
                image_name = item.attrib.get('imageName')
                vehicle_id = item.attrib.get('vehicleID')
                camera_id  = item.attrib.get('cameraID')
                pid, camid = map(int, pattern.search(image_name).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 776
                assert 1 <= camid <= 20
                camid -= 1  # index starts from 0
                img_path = osp.join(dir_path, image_name)
                if is_train:
                    pid = pid2label[pid]
                    #pid = self.dataset_name + "_" + str(pid)
                    #camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid, 0))
        else:            
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                #assert 0 <= pid <= 776
                assert 0 <= pid <= 2000
                assert 1 <= camid <= 20
                camid -= 1  # index starts from 0
                dir_path = os.path.dirname(img_path)
                base = os.path.basename(img_path)     
                name, ext = os.path.splitext(base)
                json_file = os.path.join(dir_path, name + '.json')
                if  os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        jdata = json.load(f)
                        for shape in jdata.get("shapes", []):
                            label = shape.get("label", "")
                    
                if is_train:
                    pid = pid2label[pid]
                    #pid = self.dataset_name + "_" + str(pid)
                    #camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid,0))

        return data
