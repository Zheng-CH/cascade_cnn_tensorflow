import numpy as np
from PIL import Image
import math
import time
from random import randint
import os

input_channel = 3

cali_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
cali_off_x = [-0.17, 0., 0.17]
cali_off_y = [-0.17, 0., 0.17]
cali_patt_num = len(cali_scale) * len(cali_off_x) * len(cali_off_y)
save_dir_pos_train="cali_net_data/pos_train/"
save_dir_pos_val="cali_net_data/pos_val/"
def gen_train_cali_data():
    print "Loading training db..."

    annot_dir ="/home/gavinpan/workspace/dataset/aflw/data/"
    #annot_fp = open(annot_dir + "annot", "r")
    annot_fp = open("face_rect.txt", "r")
    raw_data = annot_fp.readlines()
    annot_fp.close()    

    #pos image cropping
    x_db = [0 for _ in xrange(len(raw_data)-1)]
    for i,line in enumerate(raw_data):
      if(not line.startswith('#')):  
        print i, "/", len(raw_data), "th pos image cropping..."
        parsed_line = line.split()
        imagePath1 = parsed_line[1].strip()
        imagePath = annot_dir + imagePath1

        img = Image.open(imagePath)
        #check if not RGB
        if len(np.shape(np.asarray(img))) != 3 or np.shape(np.asarray(img))[2] != input_channel:
            continue
        
        left = int(parsed_line[2])
        upper = int(parsed_line[3])
        right = int(parsed_line[4]) + left
        lower = int(parsed_line[5]) + upper
       
        if right >= img.size[0]:
            right = img.size[0]-1
        if lower >= img.size[1]:
            lower = img.size[1]-1
        
        x_db_list = [0 for _ in xrange(cali_patt_num)]
        
        for si,s in enumerate(cali_scale):
            for xi,x in enumerate(cali_off_x):
                for yi,y in enumerate(cali_off_y):
                    #print si,xi,yi
                    new_left = left - x*float(right-left)/s
                    new_upper = upper - y*float(lower-upper)/s
                    new_right = new_left+float(right-left)/s
                    new_lower = new_upper+float(lower-upper)/s
                    
                    new_left = int(new_left)
                    new_upper = int(new_upper)
                    new_right = int(new_right)
                    new_lower = int(new_lower)


                    if new_left < 0 or new_upper < 0 or new_right >= img.size[0] or new_lower >= img.size[1]:
                        continue

                    cropped_img = img.crop((new_left, new_upper, new_right, new_lower))
                    calib_idx = si*len(cali_off_x)*len(cali_off_y)+xi*len(cali_off_y)+yi
                    #print calib_idx
                    #time.sleep(1000)
                    #for debugging

                    #print path2
                    #time.sleep(1000)
                    if calib_idx==randint(0,cali_patt_num-1):
                       path1=save_dir_pos_val + str(calib_idx)
                    else:
                       path1=save_dir_pos_train + str(calib_idx) 
                    #print path1
                    if not os.path.exists(path1):
                       os.makedirs(path1)
                    path2=path1+"/"+str(i)+".jpg"
                    cropped_img.save(path2)

if __name__ == '__main__':
    gen_train_cali_data()
