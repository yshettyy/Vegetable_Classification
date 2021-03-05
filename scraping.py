# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:58:53 2020

@author: amrut
"""
import bs4
import re
from urllib.request import urlopen as uReq
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from bs4 import BeautifulSoup as soup
import os 
import requests
import re
import codecs
# path = 'D:\\DataSet'
# os.chdir(path)


myurl = 'https://www.vegetables.co.nz/vegetables-a-z/'
uClient = uReq(myurl)
page_html = uClient.read()
uClient.close()
page_soup = soup(page_html,"html.parser")
containers = page_soup.find_all("div",{"class":"col-xs-6 col-sm-4 col-md-2 vege"})
#print(containers)

vegetables = []

#names = []
for container in containers:
    names = container.div.a.img["alt"]
    #names.append(name)
    #image = container.div.a.img["src"]
    vegetables.append(names)
    #imgs.append(image)
#print(names)
vegetables.pop(-1)

#print(vegetables)

# for veg in vegetables:
#     NewFolder = veg
#     try:
#         if not os.path.exists(NewFolder):
#             os.mkdir(NewFolder)
#     except OSError:
#         print("directory already exists")


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
final = []
for vegetable in vegetables:
    pattern = re.sub('[A-Za-z]*[-,]','',vegetable)
    final.append(pattern)
#print(final[0])
final.pop(37)
print(final)

with codecs.open('D:/Class-2/exxe.txt','w',encoding='UTF-8') as la:
    for i in final:
        la.write(i + '\n')


for i in final: 
    img = load_img(r'D:/My_DataSet/'+i+'/'+i+'.jpg') #D:\DataSet\Artichokes - globe
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    k = 0
    for batch in datagen.flow(x, batch_size = 1,save_to_dir = i,save_prefix = i ,save_format = 'jpg'):
        k+=1
        if k>30:
            break


