import numpy as np
import os
from os import listdir
from os.path import isfile, join
np.random.seed(3)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

rotateGenerator = ImageDataGenerator(rotation_range=40, fill_mode='nearest')
shiftGenerator = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, fill_mode='nearest')
shearGenerator = ImageDataGenerator(shear_range=30, fill_mode='nearest')
zoomGenerator = ImageDataGenerator( zoom_range=[-0.1,0.1], fill_mode='nearest')

rotateDataset = []
shiftDataset = []
shearDataset = []
zoomDataset = []

filename_in_dir = []

for root, dirs, files in os.walk('img_data\\img_data_r1'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        filename_in_dir.append(full_fname) #디렉토리 내의 파일 name
        print(fname)

# 144개 기존이미지 불러오기 완료
genenum=input("데이터를 몇배로 확장 하시겠습니까?(숫자 입력)")
genetype=int(input("확장방법 ? 1: rotate , 2: shear , 3: shift , 4: zoom "))

for file_image in filename_in_dir:
    img = load_img(file_image)
    print(file_image,"저장")
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i1,i2,i3,i4=0,0,0,0
    if genetype==1 :
        for batch in rotateGenerator.flow(x):
            i1 +=1
            rotateDataset.append(batch)
            if i1> int(genenum)-1:
                i1=0
                break
    if genetype == 2:
        for batch in shearGenerator.flow(x):
            i2 += 1
            shearDataset.append(batch)
            if i2 > int(genenum)-1:
                i2 = 0
                break
    if genetype == 3:
        for batch in shiftGenerator.flow(x):
            i3 += 1
            shiftDataset.append(batch)
            if i3 > int(genenum)-1:
                i3 = 0
                break
    if genetype == 4:
        for batch in zoomGenerator.flow(x):
            i4 += 1
            zoomDataset.append(batch)
            if i4 > int(genenum)-1:
                i4 = 0
                break
print(len(rotateDataset))
print(len(shearDataset))
print(len(shiftDataset))
print(len(zoomDataset))

