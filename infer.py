import os
from network import DSR_Net
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
import tensorflow as tf
from PIL import Image
import cv2, time
from function import cal_precision_recall_mae,cal_fmeasure,crf_refine


def predict(model, dataset):

    if dataset == 'DUTS-TE':
        path = '/data1/Saliency_Dataset/DUTS-TE/'
        save_path = 'result_origin/DUTS-TE/' 

    if dataset == 'DUT-OMRON':
        path = '/data1/Saliency_Dataset/DUT-OMRON/'
        save_path = 'result_origin/DUT-OMRON/' 
  
    if dataset == 'ECSSD':
        path = '/data1/Saliency_Dataset/ECSSD/'
        save_path = 'result_origin/ECSSD/' 

    if dataset == 'HKU-IS':
        path = '/data1/Saliency_Dataset/HKU-IS/'
        save_path = 'result_origin/HKU-IS/' 

    if dataset == 'PASCAL-S':
        path = '/data1/Saliency_Dataset/PASCAL-S/'
        save_path = 'result_origin/PASCAL-S/'


    names = os.listdir(path+'Img')

    mean = np.asarray([0.485, 0.456, 0.406]).reshape([1,1,3])
    std = np.asarray([0.229, 0.224, 0.225]).reshape([1,1,3])

    for name in names:

        img = Image.open(path+'Img/'+name).convert('RGB')

        h,w = img.size 
        th,tw = 432,432
        img = img.resize((tw, th), Image.BILINEAR)

        img = np.asarray(img)/255.
        img = (img - mean)/std

        X = np.expand_dims(img,axis=0)

        predict = model.predict(X,batch_size=1)
            
        predict = np.squeeze(predict)
        predict = zoom(predict,[w/(tw*1.0),h/(th*1.0)])
        predict = predict * 255

        cv2.imwrite(save_path+name[:-4]+'.png',predict)

def crf(dataset):

    if dataset == 'DUTS-TE':
        img_path = '/data1/Saliency_Dataset/DUTS-TE/'
        annos_path = 'result_origin/DUTS-TE/'
        save_path = 'result/DUTS-TE/'

    if dataset == 'DUT-OMRON':
        img_path = '/data1/Saliency_Dataset/DUT-OMRON/'
        annos_path = 'result_origin/DUT-OMRON/'
        save_path = 'result/DUT-OMRON/'

    if dataset == 'ECSSD':
        img_path = '/data1/Saliency_Dataset/ECSSD/'
        annos_path = 'result_origin/ECSSD/'
        save_path = 'result/ECSSD/'

    if dataset == 'HKU-IS':
        img_path = '/data1/Saliency_Dataset/HKU-IS/'
        annos_path = 'result_origin/HKU-IS/'
        save_path = 'result/HKU-IS/'

    if dataset == 'PASCAL-S':
        img_path = '/data1/Saliency_Dataset/PASCAL-S/'
        annos_path = 'result_origin/PASCAL-S/'
        save_path = 'result/PASCAL-S/'

    names = os.listdir(img_path+'Img')

    for name in names:

        img = cv2.imread(img_path+'Img/'+name, 1)
        annos = cv2.imread(annos_path+name[:-4]+'.png', 0)
        crf_img = crf_refine(img, annos)
        cv2.imwrite(save_path+name[:-4]+'.png',crf_img)

def eval(dataset, crf):

    if dataset == 'DUTS-TE':
        GT_path = '/data1/Saliency_Dataset/DUTS-TE/'
        if crf:
            annos_path = 'result/DUTS-TE/'
        else:
            annos_path = 'result_origin/DUTS-TE/'

    if dataset == 'DUT-OMRON':
        GT_path = '/data1/Saliency_Dataset/DUT-OMRON/'
        if crf:
            annos_path = 'result/DUT-OMRON/'
        else:
            annos_path = 'result_origin/DUT-OMRON/'

    if dataset == 'ECSSD':
        GT_path = '/data1/Saliency_Dataset/ECSSD/'
        if crf:
            annos_path = 'result/ECSSD/'
        else:
            annos_path = 'result_origin/ECSSD/'

    if dataset == 'HKU-IS':
        GT_path = '/data1/Saliency_Dataset/HKU-IS/'
        if crf:
            annos_path = 'result/HKU-IS/'
        else:
            annos_path = 'result_origin/HKU-IS/'

    if dataset == 'PASCAL-S':
        GT_path = '/data1/Saliency_Dataset/PASCAL-S/'
        if crf:
            annos_path = 'result/PASCAL-S/'
        else:
            annos_path = 'result_origin/PASCAL-S/'


    names = os.listdir(GT_path+'GT')
    maes = []
    fmeasures = []

    for name in names:

        annos = np.asarray(Image.open(annos_path+name).convert('L'))
        GT = np.asarray(Image.open(GT_path+'GT/'+name).convert('L'))

        precision, recall, mae = cal_precision_recall_mae(annos,GT)
        fmeasure = cal_fmeasure(precision, recall)

        maes.append(mae)
        fmeasures.append(fmeasure)

    maes = np.asarray(maes)
    fmeasures = np.asarray(fmeasures)
    print dataset,'Mae:',maes.mean(),'Fmeasure:',fmeasures.mean()

if __name__ == '__main__':
    model = DSR_Net((432, 432, 3))
    model.load_weights('model/model-final.hdf5')

    predict_list = ['PASCAL-S', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON', 'ECSSD']
    for dataset in predict_list:
        predict(model, dataset)
        crf(dataset)
        eval(dataset, True)#True indicates CRF postprocess


