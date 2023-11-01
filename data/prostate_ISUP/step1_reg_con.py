import SimpleITK as sitk
import os
import numpy as np
import cv2
import copy


def align_seg_with_raw_nrrd(dcm, seg):
    # Just for labelmap .... because of nearestNeighour interpolator
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(dcm)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_new = resampler.Execute(seg)
    return seg_new


def normalize(img_):
    #img_[img_>=250] = 250
    #img_[img_<=-200] = -200
        
    
    
    # mean = np.mean(img_)
    # std = np.std(img_)
    # mean = -0.29790374886599763
    # std = 0.29469745653088375
    # img_ = (img_ - mean) / std
    max_ = np.max(img_)
    min_ = np.min(img_)
    img_ = (img_ - min_) / (max_ - min_ + 1e-9)
    img_ = img_ * 2 - 1
    return img_

train_file_path = './Case_lesion12345/'

for dir, file, images in os.walk(train_file_path):
    if images != [] and dir.split('/')[-1].split('_')[-1] =='T2W' :

        ID = dir.split('/')[-1].split('_')[0] + '_T2W.nii'
        
        gt_path = dir + '/' + ID
        gt_sitk = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt_sitk)
        gt[gt!=1]=0

        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dir, series_id[0])
        # print(len(series_file_names))
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3d_T2W = series_reader.Execute() 
        Images = sitk.GetArrayFromImage(image3d_T2W)
        
        
        originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
        mask = np.zeros((Images.shape[0],224,224))
        mask_all = np.zeros((Images.shape[0],224,224))
        Images = normalize(Images)
        #originGT = np.zeros((Images.shape[0],256,256)).astype(np.float32)
        for i in range(len(Images)):
            img = Images[i]
            mask_ = np.zeros((gt.shape[1],gt.shape[2]))
            if np.max(gt[i])>0:
                slices_mask = gt[i]
                slices_mask = slices_mask[:,:,np.newaxis]
                # slices_mask = np.concatenate((slices_mask, slices_mask, slices_mask), axis = 2)
                
                _, binaryzation = cv2.threshold(slices_mask, 0, 255, cv2.THRESH_BINARY)
    
                contours, _ = cv2.findContours(np.uint8(binaryzation), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                area = []
                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))
                mask_ = cv2.drawContours(mask_, contours, max_idx, 1, cv2.FILLED)
                mask_[gt[i]>=2] = 2
                mask_ = cv2.resize(mask_, (224, 224),cv2.INTER_NEAREST)
            else:
                mask_ = np.zeros((224,224))
            #img = normalize(img)
            
            img = cv2.resize(img, (224, 224))
            mask[i] = mask_
            originImg[i] = img
        
        fold_ID = './processed_data/'+dir.split('/')[-1].split('_')[0]
        print(fold_ID)
        if not os.path.exists(fold_ID):  
            os.makedirs(fold_ID)
        resultImage_ = sitk.GetImageFromArray(originImg)  
        sitk.WriteImage(resultImage_, fold_ID+'/T2W.nii.gz')
        mask_PZ = copy.copy(mask)
        
        
        gt_sitk = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt_sitk)
        gt[gt<2]=0
        # gt[gt>=2]=1
        for i in range(len(Images)):
            img = Images[i]
            mask_ = np.zeros((gt.shape[1],gt.shape[2]))
            if np.max(gt[i])>0:
                slices_mask = gt[i]
                slices_mask = slices_mask[:,:,np.newaxis]
                # slices_mask = np.concatenate((slices_mask, slices_mask, slices_mask), axis = 2)
                
                _, binaryzation = cv2.threshold(slices_mask, 0, 255, cv2.THRESH_BINARY)
    
                contours, _ = cv2.findContours(np.uint8(binaryzation), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                area = []
                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))
                mask_ = cv2.drawContours(mask_, contours, max_idx, 1, cv2.FILLED)
                # mask_[gt[i]==1] = 1
                mask_[gt[i]==2] = 2
                mask_[gt[i]==3] = 3
                mask_[gt[i]==4] = 4
                mask_[gt[i]==5] = 5
                mask_ = cv2.resize(mask_, (224, 224))
            else:
                mask_ = np.zeros((224,224))
            #img = normalize(img)
            
            img = cv2.resize(img, (224, 224))
            # mask[i][mask_>0.2] = 1
            mask[i][mask_>1.2] = 2
            mask[i][mask_>2.2] = 3
            mask[i][mask_>3.2] = 4
            mask[i][mask_>4.2] = 5

            originImg[i] = img
        mask_all = copy.copy(mask)
        resultImage_ = sitk.GetImageFromArray(mask)  
        
        sitk.WriteImage(resultImage_, fold_ID+'/T2W_gt.nii.gz')
        
        
        dir_DWI = dir.split('\\')[0] + '/' + dir.split('/')[-1].split('_')[0] + '_DWI'
        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir_DWI)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dir_DWI, series_id[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3d = series_reader.Execute()
        image3d = align_seg_with_raw_nrrd(image3d_T2W, image3d)
        Images = sitk.GetArrayFromImage(image3d)
        originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
        mask = np.zeros((Images.shape[0],224,224))
        Images = normalize(Images)
        try:
            gt_sitk = sitk.ReadImage(dir_DWI+'/'+dir.split('/')[-1].split('_')[0] + '_DWI.nii')
            gt_sitk = align_seg_with_raw_nrrd(image3d_T2W, gt_sitk)
            gt = sitk.GetArrayFromImage(gt_sitk)
            gt[gt==1] = 0
        except:
            print("Not exists:", dir_DWI+'/'+dir.split('/')[-1].split('_')[0] + '_DWI.nii.gz')
            gt = np.zeros(mask.shape)
        for i in range(len(Images)):
            img = Images[i]
            mask_ = np.zeros((gt.shape[1],gt.shape[2]))
            if np.max(gt[i])>0:
                slices_mask = gt[i]
                slices_mask = slices_mask[:,:,np.newaxis]
                # slices_mask = np.concatenate((slices_mask, slices_mask, slices_mask), axis = 2)
                
                _, binaryzation = cv2.threshold(slices_mask, 0, 255, cv2.THRESH_BINARY)
    
                contours, _ = cv2.findContours(np.uint8(binaryzation), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                area = []
                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))
                #mask_ = cv2.drawContours(mask_, contours, max_idx, 1, cv2.FILLED)
                mask_[gt[i]==2] = 2
                mask_[gt[i]==3] = 3
                mask_[gt[i]==4] = 4
                mask_[gt[i]==5] = 5
                mask_ = cv2.resize(mask_, (224, 224))
            else:
                mask_ = np.zeros((224,224))
            #img = normalize(img)
            
            img = cv2.resize(img, (224, 224))
            mask[i][mask_>1.2] = 2
            mask[i][mask_>2.2] = 3
            mask[i][mask_>3.2] = 4
            mask[i][mask_>4.2] = 5
            originImg[i] = img
        mask_all[mask==2] = 2
        mask_all[mask==3] = 3
        mask_all[mask==4] = 4
        mask_all[mask==5] = 5
        resultImage_ = sitk.GetImageFromArray(mask)  
        sitk.WriteImage(resultImage_, fold_ID+'/DWI_gt.nii.gz')
        
        
        resultImage_ = sitk.GetImageFromArray(originImg)  
        sitk.WriteImage(resultImage_, fold_ID+'/DWI.nii.gz')
        
        
        ID = dir.split('/')[-1].split('_')[0] + '_T2W_PZ.nii.gz'
        
        gt_path = dir + '/' + ID
        gt_sitk = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt_sitk)
        gt[gt!=1]=0

        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dir, series_id[0])
        print(len(series_file_names))
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3d = series_reader.Execute()
   
        Images = sitk.GetArrayFromImage(image3d)
        originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
        mask = np.zeros((Images.shape[0],224,224))
        Images = normalize(Images)
        #originGT = np.zeros((Images.shape[0],256,256)).astype(np.float32)
        for i in range(len(Images)):
            img = Images[i]
            mask_ = np.zeros((gt.shape[1],gt.shape[2]))
            if np.max(gt[i])>0:
                slices_mask = gt[i]
                slices_mask = slices_mask[:,:,np.newaxis]
                # slices_mask = np.concatenate((slices_mask, slices_mask, slices_mask), axis = 2)
                
                _, binaryzation = cv2.threshold(slices_mask, 0, 255, cv2.THRESH_BINARY)
    
                contours, _ = cv2.findContours(np.uint8(binaryzation), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                area = []
                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))
                mask_ = cv2.drawContours(mask_, contours, max_idx, 1, cv2.FILLED)
                mask_[gt[i]==1] = 1
                mask_ = cv2.resize(mask_, (224, 224),cv2.INTER_NEAREST)
            else:
                mask_ = np.zeros((224,224))
            #img = normalize(img)
            
            img = cv2.resize(img, (224, 224))
            mask[i] = mask_
            originImg[i] = img
        mask_PZ[mask==1] = 2
        fold_ID = './processed_data/'+dir.split('/')[-1].split('_')[0]
        print(fold_ID)
        if not os.path.exists(fold_ID):  
            os.makedirs(fold_ID)
        resultImage_ = sitk.GetImageFromArray(mask_PZ)  
        sitk.WriteImage(resultImage_, fold_ID+'/T2W_PZ.nii.gz')
        
        dir_ADC = dir.split('\\')[0] + '/' + dir.split('/')[-1].split('_')[0] + '_ADC'
        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir_ADC)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dir_ADC, series_id[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3d = series_reader.Execute()
        image3d = align_seg_with_raw_nrrd(image3d_T2W, image3d)
        Images = sitk.GetArrayFromImage(image3d)
        originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
        mask = np.zeros((Images.shape[0],224,224))
        Images = normalize(Images)
        try:
            gt_sitk = sitk.ReadImage(dir_ADC+'/'+dir.split('/')[-1].split('_')[0] + '_ADC.nii')
            gt_sitk = align_seg_with_raw_nrrd(image3d_T2W, gt_sitk)
            gt = sitk.GetArrayFromImage(gt_sitk)
            gt[gt==1] = 0
        except:
            print("Not exists:", dir_ADC+'/'+dir.split('/')[-1].split('_')[0] + '_ADC.nii.gz')
            gt = np.zeros(mask.shape)
        for i in range(len(Images)):
            img = Images[i]
            mask_ = np.zeros((gt.shape[1],gt.shape[2]))
            if np.max(gt[i])>0:
                slices_mask = gt[i]
                slices_mask = slices_mask[:,:,np.newaxis]
                # slices_mask = np.concatenate((slices_mask, slices_mask, slices_mask), axis = 2)
                
                _, binaryzation = cv2.threshold(slices_mask, 0, 255, cv2.THRESH_BINARY)
    
                contours, _ = cv2.findContours(np.uint8(binaryzation), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                area = []
                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))
                mask_ = cv2.drawContours(mask_, contours, max_idx, 1, cv2.FILLED)
                mask_[gt[i]==2] = 2
                mask_[gt[i]==3] = 3
                mask_[gt[i]==4] = 4
                mask_[gt[i]==5] = 5
                mask_ = cv2.resize(mask_, (224, 224))
            else:
                mask_ = np.zeros((224,224))
            #img = normalize(img)
            
            img = cv2.resize(img, (224, 224))
            mask[i][mask_>1.2] = 2
            mask[i][mask_>2.2] = 3
            mask[i][mask_>3.2] = 4
            mask[i][mask_>4.2] = 5
            originImg[i] = img
        
        
        mask_all[mask==2] = 2
        mask_all[mask==3] = 3
        mask_all[mask==4] = 4
        mask_all[mask==5] = 5
        
        resultImage_ = sitk.GetImageFromArray(originImg)  
        sitk.WriteImage(resultImage_, fold_ID+'/ADC.nii.gz')
        resultImage_ = sitk.GetImageFromArray(mask)  
        sitk.WriteImage(resultImage_, fold_ID+'/ADC_gt.nii.gz')
        resultImage_ = sitk.GetImageFromArray(mask_all)  
        sitk.WriteImage(resultImage_, fold_ID+'/Con_gt.nii.gz')
        # resultImage_ = sitk.GetImageFromArray(mask)  
        # sitk.WriteImage(resultImage_, dir+'/gt.nii')
        
        
        
        resultImage = sitk.GetImageFromArray(Images)  
        sitk.WriteImage(resultImage, './ADC.nii.gz')

            


