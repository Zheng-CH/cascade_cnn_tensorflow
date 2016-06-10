import os 
import cv2
import numpy as np
import time
from random import randint

input_channel = 3
img_size_12 = 12
img_size_24 = 24
img_size_48 = 48
dim_12 = img_size_12 * img_size_12 * input_channel
dim_24 = img_size_24 * img_size_24 * input_channel
dim_48 = img_size_48 * img_size_48 * input_channel

neg_per_img = 35
face_minimum = 27
thr=1000

pos_dir="./pos/"
save_dir_pos_12="./pos_12_train/"
save_dir_pos_12_val="./pos_12_val/"
save_dir_pos_24="./pos_24_train/"
save_dir_pos_24_val="./pos_24_val/"
save_dir_pos_48="./pos_48_train/"
save_dir_pos_48_val="./pos_48_val/"

neg_dir="./bg1/"
save_dir_neg_12="./neg_12_train/"
save_dir_neg_12_val="./neg_12_val/"
save_dir_neg_24="./neg_24/"
save_dir_neg_48="./neg_48/"

save_dir_pos="./net_data/pos_train/"
save_dir_pos_val="./net_data/pos_val/"
save_dir_neg="./net_data/neg_train/"
save_dir_neg_val="./net_data/neg_val/"

def img2array(img,dim):   
    if dim == img_size_12:    
        if img.shape[0] != img_size_12 or img.shape[1] != img_size_12:
            #img = img.resize((img_size_12,img_size_12))
			img = cv2.resize(img,(img_size_12,img_size_12))            
        #img = np.asarray(img).astype(np.float32)/255 
        #img = np.reshape(img,(1,dim_12))
    elif dim == img_size_24:
        if img.shape[0] != img_size_24 or img.shape[1] != img_size_24:
            #img = img.resize((img_size_24,img_size_24))
			img = cv2.resize(img,(img_size_24,img_size_24))
        #img = np.asarray(img).astype(np.float32)/255
        #img = np.reshape(img,(1,dim_24))
    elif dim == img_size_48:
        if img.shape[0] != img_size_48 or img.shape[1] != img_size_48:
            #img = img.resize((img_size_48,img_size_48))
			img = cv2.resize(img,(img_size_48,img_size_48))
        #img = np.asarray(img).astype(np.float32)/255
        #img = np.reshape(img,(1,dim_48))
    return img


def gen_12_net_pos_data(pos_dir,save_dir_pos_12,save_dir_pos_12_val,thr):
    num=0
    for root,dirs,files in os.walk(pos_dir):
       #print root
       for file in files:      
           path1=os.path.join(root,file)
           #print path1
           img1 = cv2.imread(path1)
           #print img1.shape
           img2=img2array(img1,img_size_12)
           #print img2.shape
           num+=1
           if(num>thr):
              path2=save_dir_pos_12+file[:-4]+"_12.jpg";
              cv2.imwrite(path2,img2);
           else:
              path2=save_dir_pos_12_val+file[:-4]+"_12.jpg";
              cv2.imwrite(path2,img2);
           #time.sleep(2)
           print num

def gen_24_net_pos_data(pos_dir,save_dir_pos_24,save_dir_pos_24_val,thr):
    num=0
    for root,dirs,files in os.walk(pos_dir):
       #print root
       for file in files:      
           path1=os.path.join(root,file)
           #print path1
           img1 = cv2.imread(path1)
           #print img1.shape
           img2=img2array(img1,img_size_24)
           #print img2.shape
           num+=1
           if(num>thr):
              path2=save_dir_pos_24+file[:-4]+"_24.jpg";
              cv2.imwrite(path2,img2);
           else:
              path2=save_dir_pos_24_val+file[:-4]+"_24.jpg";
              cv2.imwrite(path2,img2);
           #time.sleep(2)
           print num

def gen_48_net_pos_data(pos_dir,save_dir_pos_48,save_dir_pos_48_val,thr):
    num=0
    for root,dirs,files in os.walk(pos_dir):
       #print root
       for file in files:      
           path1=os.path.join(root,file)
           #print path1
           img1 = cv2.imread(path1)
           #print img1.shape
           img2=img2array(img1,img_size_48)
           #print img2.shape
           num+=1
           if(num>thr):
              path2=save_dir_pos_48+file[:-4]+"_48.jpg";
              cv2.imwrite(path2,img2);
           else:
              path2=save_dir_pos_48_val+file[:-4]+"_48.jpg";
              cv2.imwrite(path2,img2);
           #time.sleep(2)
           print num

def gen_net_neg_data(neg_dir,save_dir_neg,save_dir_neg_val,thr):
    num=0
    for root,dirs,files in os.walk(neg_dir):
       #print root
       for file in files:
           path1=os.path.join(root,file)
           #print path1
           img1 = cv2.imread(path1)
           #print img1.shape
           num+=1
           for neg_iter in xrange(neg_per_img):
               rad_rand = randint(0,min(img1.shape[0],img1.shape[1])-1)
               while(rad_rand <= face_minimum):
				   rad_rand = randint(0,min(img1.shape[0],img1.shape[1])-1)

               x_rand = randint(0, img1.shape[1] - rad_rand - 1)
               y_rand = randint(0, img1.shape[0] - rad_rand - 1)
               
               neg_cropped_img = img1[y_rand:y_rand + rad_rand,x_rand:x_rand + rad_rand]
               
               if((num<=thr)&(neg_iter==0)):
                   cv2.imwrite(save_dir_neg_val + file[:-4]+"_"+str(neg_iter) + ".jpg", neg_cropped_img)
               else:
                   cv2.imwrite(save_dir_neg + file[:-4]+"_"+str(neg_iter) + ".jpg", neg_cropped_img)
           
           print num

def gen_net_pos_data(pos_dir,save_dir_pos,save_dir_pos_val,thr):
    num=0
    for root,dirs,files in os.walk(pos_dir):
       #print root
       for file in files:      
           path1=os.path.join(root,file)
           #print path1
           img1 = cv2.imread(path1)
           #print img1.shape
           num+=1
           if(num>thr):
              path2=save_dir_pos+file;
              cv2.imwrite(path2,img1);
           else:
              path2=save_dir_pos_val+file;
              cv2.imwrite(path2,img1);
           #time.sleep(2)
           print num
if __name__ == '__main__':
    #gen pos data 
    #gen_12_net_pos_data(pos_dir,save_dir_pos_12,save_dir_pos_12_val,thr)
    #gen_24_net_pos_data(pos_dir,save_dir_pos_24,save_dir_pos_24_val,thr)
    #gen_48_net_pos_data(pos_dir,save_dir_pos_48,save_dir_pos_48_val,thr)
    
    #gen neg data
    gen_net_pos_data(pos_dir,save_dir_pos,save_dir_pos_val,thr)
    gen_net_neg_data(neg_dir,save_dir_neg,save_dir_neg_val,thr)
