import os
import random
import shutil

IMAGE_PATH = 'worskpace/training_demo/CHVG-Dataset/'
SPLITS = (80, 20)

# print(os.listdir('../../'))

images = os.listdir(IMAGE_PATH)

names = []
train = []
test = []

for i in images:
    name = i[:-3]
    if i[-3:] == 'xml':
        names.append(name)

random.shuffle(names)

train_size = int(len(names)*SPLITS[0]/(sum(SPLITS)))

train = names[:train_size]
test = names[train_size:]

for i in train:
    shutil.copyfile(IMAGE_PATH+i+'xml', IMAGE_PATH+'train/'+i+'xml')
    shutil.copyfile(IMAGE_PATH+i+'jpg', IMAGE_PATH+'train/'+i+'jpg')

for i in test:
    shutil.copyfile(IMAGE_PATH+i+'xml', IMAGE_PATH+'test/'+i+'xml')
    shutil.copyfile(IMAGE_PATH+i+'jpg', IMAGE_PATH+'test/'+i+'jpg')