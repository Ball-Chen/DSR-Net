from PIL import Image
import numpy as np
import os,random
import cv2
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
import time,threading


def RandomHorizontallyFlip(img, mask):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def random_crop(img, mask, size):
    assert img.size == mask.size
    w, h = img.size
    th, tw = size,size
    if w == tw and h == th:
        return img, mask
    if w < tw or h < th:
        return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

def RandomRotate(img, mask, degree):
    rotate_degree = random.random() * 2 * degree - degree
    return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


def data_generator(path,batch_size):

    names = os.listdir(path+'Img/')

    X = np.zeros([batch_size, 384, 384, 3])
    Y = np.zeros([batch_size, 384, 384, 1])

    mean = np.asarray([0.485, 0.456, 0.406]).reshape([1,1,3])
    std = np.asarray([0.229, 0.224, 0.225]).reshape([1,1,3])

    i = 0
    while 1:
 
        random.shuffle(names)
        random.shuffle(names)
        random.shuffle(names)
        for name in names:

            img = Image.open(path+'Img/'+name).convert('RGB')
            mask = Image.open(path+'GT/'+name[:-4]+'.png').convert('L')

            img, mask = RandomHorizontallyFlip(img, mask)
            img, mask = random_crop(img, mask, 384)
            img, mask = RandomRotate(img, mask, 10)
            
            img = np.asarray(img)/255.
            mask = np.asarray(mask)/255.

            img = (img - mean)/std

            X[i] = img
            Y[i,:,:,0] = np.round(mask)
            
            i = i + 1
            if i >= batch_size:
                i = 0
                yield {'input': X}, {'output':Y}
        
            

class multi_threads_generator(object):
    def __init__(self,path, batch_size):
        self.lock = threading.Lock()
        self.generator = data_generator(path, batch_size)

    def next(self):
        with self.lock:
            return self.generator.next()


