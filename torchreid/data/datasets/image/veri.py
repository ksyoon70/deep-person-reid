# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os
import os.path as osp
import re
from lxml import etree
from torchreid.utils import read_image
from ..dataset import ImageDataset

class VeRi(ImageDataset):
    """VeRi.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "veri"
    dataset_name = "veri"

    train_xml_path = "train_label.xml"
    test_xml_path = "test_label.xml"


    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self.train_xml_file = osp.join(self.dataset_dir, self.train_xml_path)
        self.test_xml_file = osp.join(self.dataset_dir, self.test_xml_path)
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
        gallery = self.process_dir(self.gallery_dir, self.test_xml_file, is_train=True)
        
        super(VeRi, self).__init__(train, query, gallery, **kwargs)
        #추가로 color_id 와 type_id 갯수를 계산을 한다.
        self.num_train_color_ids = self.get_num_colors(train)
        self.num_train_type_ids = self.get_num_types(train)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid, color_id, type_id = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid,
            'color_id': color_id,
            'type_id': type_id
        }
        return item

    def get_num_colors(self, data):
        """Returns the number of training colors.

        Each tuple in data contains (img_path(s), pid, camid, dsetid, color_id, type_id).
        """
        colors = set()
        for items in data:
            colorid = items[4]
            colors.add(colorid)
        return len(colors)

    def get_num_types(self, data):
        """Returns the number of training types.

        Each tuple in data contains (img_path(s), pid, camid, dsetid, color_id, type_id).
        """
        types = set()
        for items in data:
            typeid = items[5]
            types.add(typeid)
        return len(types)    

    def process_dir(self, dir_path, xml_file, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        #주의 veri 의 데이터셋에서 주의 할 점은 파일의 pid 갯수와 xml 파일의 pid 갯수가 다르다는 것이다.
        # 예를 들어 파일의 pid 갯수는 576개이고 xml 파일의 pid 갯수는575개이다.
        pid_container = set()

        if not xml_file:
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
                color_id   = item.attrib.get('colorID')
                type_id    = item.attrib.get('typeID')
                pid, camid = map(int, pattern.search(image_name).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 776
                assert 1 <= camid <= 20
                camid -= 1  # index starts from 0
                color_id = int(color_id) - 1
                type_id = int(type_id) - 1
                img_path = osp.join(dir_path, image_name)
                if is_train:
                    pid = pid2label[pid]
                    #pid = self.dataset_name + "_" + str(pid)
                    #camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid, 0, color_id, type_id))
        else:
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 776
                assert 1 <= camid <= 20
                camid -= 1  # index starts from 0
                if is_train:
                    pid = pid2label[pid]
                    #pid = self.dataset_name + "_" + str(pid)
                    #camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid,0,0,0))
        
        return data
        
    
