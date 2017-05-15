import numpy as np
import cv2
import os
import re
import random
import math

import utils


labels = np.loadtxt("./labels/train_labels.txt")
print labels


class Reader():
  def __init__(self, config):
    self.imgnames = list()
    self.labels = np.loadtxt(config.labels_file)

    with open(config.trainset_file, 'rb') as fr:
      for line in fr:
        line = line.strip()
        self.imgnames.append(line)
    
    self.size = len(self.imgnames)
    self.batch_size = config.batch_size
    self.imgs_path = config.imgs_path
    self.lineidx = 0

  def random_batch(self):
    rand = random.sample(xrange(self.size), self.batch_size)
#    print rand
    batch_imgnames = list()
    for idx in rand:
       batch_imgnames.append(self.imgnames[idx])
    batch_labels = labels[rand]

    img_list = list()
    for imgname in batch_imgnames:
      path = self.imgs_path+imgname+".jpg"
    #  print path
      #img = cv2.imread(path)
      #img = cv2.resize(img, (224, 224))
      img = utils.load_image(path)
      img_list.append(img)

    #print batch_labels
    #print img_list
    batch_imgs = np.reshape(np.stack(img_list),[-1,224,224,3])
    batch_labels = np.reshape(batch_labels, [-1, 20])
    return batch_imgs, batch_labels

  def batch(self):
    batch_imgnames = list()
    for idx in range(self.lineidx, self.lineidx+self.batch_size):
       batch_imgnames.append(self.imgnames[idx])
    #batch_labels = labels[self.lineidx:(self.lineidx+self.batch_size)]
    batch_labels = self.labels[self.lineidx:(self.lineidx+self.batch_size)]
    self.lineidx = self.lineidx+self.batch_size

    img_list = list()
    for imgname in batch_imgnames:
      path = self.imgs_path+imgname+".jpg"
    #  print path
      #img = cv2.imread(path)
      #img = cv2.resize(img, (224, 224))
      img = utils.load_image(path)
      img_list.append(img)

    #print batch_labels
    #print img_list
    batch_imgs = np.reshape(np.stack(img_list),[-1,224,224,3])
    batch_labels = np.reshape(batch_labels, [-1, 20])
    return batch_imgs, batch_labels
    
def main():
  reader = Reader("./labels/train_image.txt","./labels/train_labels.txt","./JPEGImages/")
  imgs, labels = reader.random_batch()
  print imgs.shape, type(imgs), labels.shape, type(labels)
    
if __name__ == "__main__":
  main()
