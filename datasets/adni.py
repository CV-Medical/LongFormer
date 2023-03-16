import torch
import torch.utils.data

import os

import random

import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F


'''
label

0 NC
1 AD
2 sMCI
3 pMCI

'''

class ADNIDataset:
    def __init__(self, img_folder, num_classes, classification_type, num_visits=3, image_set='train'):
        self.img_folder = img_folder
        self.num_classes = num_classes
        self.num_visits = num_visits
        self.classification_type = classification_type
        csv_path = os.path.join(img_folder, 'annotations', '{}.csv'.format(image_set))
        self.image_set = image_set
        print('Constructing ADNI...')
        self.getPatientsInfo(csv_path)

    def getPatientsInfo(self,csv_path):
        if not os.path.exists(csv_path):
            raise ValueError("{} dir not found".format(csv_path))

        self._path_to_patients = []
        self._labels = []
        self._num_visits = []
        with open(csv_path) as f:
            for line in f:
                patient_name, label, num_visits = line.split('\n')[0].split(' ')
                patient_dir = os.path.join(self.img_folder, 'images', patient_name)
                if int(num_visits) < self.num_visits:
                    continue

                if self.classification_type == 'NC/AD':
                    if int(label) not in [0,1]:
                        continue
                elif self.classification_type == 'sMCI/pMCI':
                    if int(label) not in [2,3]:
                        continue
                 
                self._path_to_patients.append(patient_dir)
                self._labels.append(int(label))
                self._num_visits.append(int(num_visits))
    
        

        print("Constructing adni patients (size: {}) from {}".format(len(self._path_to_patients), csv_path))

    def __len__(self):
        return len(self._path_to_patients)

    def __getitem__(self, idx):

        instance_check = False
        while not instance_check:          
            sequence_label = self._labels[idx]
            
            patient_dir = self._path_to_patients[idx]
            patient_id = patient_dir.split('/')[-1]

            imgs = []
            boxes = []

            visits = os.listdir(patient_dir)
            if '.DS_Store' in visits:
                visits.remove('.DS_Store')
            visits = sorted(visits)
            
            for visit_idx in range(self.num_visits):
                visit = visits[int(visit_idx)]
                nii_path = os.path.join(patient_dir, visit, 't1.nii.gz')
                itk_img = sitk.ReadImage(nii_path)
                img = torch.as_tensor(sitk.GetArrayFromImage(itk_img))

                img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear', align_corners=True).squeeze()
                
                imgs.append(torch.as_tensor(img,dtype=torch.float32))

                x_list,y_list,z_list = np.where(img!=0)
                    
                x1, y1, z1 = min(x_list), min(y_list), min(z_list)
                x2, y2, z2 = max(x_list), max(y_list), max(z_list)
                    
                x_c = (x2+x1)/2
                y_c = (y2+y1)/2
                z_c = (z2+z1)/2
                d_i = x2-x1
                w_i = y2-y1
                h_i = z2-z1

                boxes.append((x_c/img.shape[0],y_c/img.shape[1],z_c/img.shape[2],d_i/img.shape[0],w_i/img.shape[1],h_i/img.shape[2]))

            target = {
                'patient_id': torch.as_tensor(int(patient_id.split('_')[-1])), 
                'label': torch.as_tensor([sequence_label]).long(),
                'boxes': torch.as_tensor(boxes, dtype=torch.float32).unsqueeze(1),
                }

            instance_check=True
            

        return torch.stack(imgs), target 


def build(image_set, args):
    
    if args.dataset_file == 'ADNI':

        print('use ADNI dataset')
        img_folder = args.data_path
        dataset = ADNIDataset(img_folder, args.num_classes, args.classification_type, num_visits=args.num_visits, image_set=image_set)

    return dataset
