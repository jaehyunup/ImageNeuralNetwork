import numpy as np
import os
from os import listdir
from os.path import isfile, join
np.random.seed(3)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL

rotateGenerator = ImageDataGenerator(rotation_range=40, fill_mode='nearest')
shiftGenerator = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, fill_mode='nearest')
shearGenerator = ImageDataGenerator( shear_range=30, fill_mode='nearest')
zoomGenerator = ImageDataGenerator( zoom_range=[-0.1,0.1], fill_mode='nearest')
filename_in_dir = []

for root, dirs, files in os.walk('img_data\\img_data_r1'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        filename_in_dir.append(full_fname) #디렉토리 내의 파일 name
        print(fname)
a=0

for file_image in filename_in_dir:
    img = load_img(file_image)
    print(file_image,"저장")
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    os.makedirs('img_data\\img_data_rotate\\frame0000%d'%a)
    os.makedirs('img_data\\img_data_shear\\frame0000%d' % a)
    os.makedirs('img_data\\img_data_shift\\frame0000%d' % a)
    os.makedirs('img_data\\img_data_zoom\\frame0000%d' % a)
    i1,i2,i3,i4=0,0,0,0
    for batch in rotateGenerator.flow(x, save_to_dir='img_data\\img_data_rotate\\frame0000%d' % a,save_prefix='frame0000%d' % a, save_format='jpg'):
        i1 +=1
        if i1>9 :
            i1=0
            break
    for batch in shiftGenerator.flow(x, save_to_dir='img_data\\img_data_shear\\frame0000%d' % a,save_prefix='frame0000%d' % a, save_format='jpg'):
        i2 += 1
        if i2 > 9:
            i2 = 0
            break
    for batch in shearGenerator.flow(x, save_to_dir='img_data\\img_data_shift\\frame0000%d' % a,save_prefix='frame0000%d' % a, save_format='jpg'):
        i3 += 1
        if i3 > 9:
            i3 = 0
            break
    for batch in zoomGenerator.flow(x, save_to_dir='img_data\\img_data_zoom\\frame0000%d' % a,save_prefix='frame0000%d' % a, save_format='jpg'):
        i4 += 1
        if i4 > 9:
            i4 = 0
            break
    a += 1