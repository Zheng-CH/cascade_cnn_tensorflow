#!/usr/bin/env python2
import numpy as np
from PIL import Image
import math
import time
from random import randint
import os
from skimage.transform import pyramid_gaussian
from compiler.ast import flatten

import etc

def load_db_detect_train():
   
    print "Loading training db..."

    #annot_dir = etc.db_dir + "aflw/data/"
    annot_dir ="/home/gavinpan/workspace/dataset/aflw/data/"
    #annot_fp = open(annot_dir + "annot", "r")
    annot_fp = open("face_rect.txt", "r")
    raw_data = annot_fp.readlines()
    annot_fp.close()
    #pos image cropping
    print "pos data number={}".format(len(raw_data)-1)
    pos_db = [0 for _ in xrange(len(raw_data)-1)]
    for i,line in enumerate(raw_data):
      	if(not line.startswith('#')):
            if(i%5000==0):
                print i, "/", len(raw_data), "th pos image cropping..."
            parsed_line = line.split()
            imagePath = parsed_line[1].strip()
            imagePath = annot_dir + imagePath
            img = Image.open(imagePath)
            #check if not RGB
            if len(np.shape(np.asarray(img))) != 3 or np.shape(np.asarray(img))[2] != etc.input_channel:
                continue
            pos_db_line = np.zeros((2,etc.dim_12 + etc.dim_24 + etc.dim_48), np.float32)
            #left = int(parsed_line[1])
            #upper = int(parsed_line[2])
            #right = int(parsed_line[3]) + left
            #lower = int(parsed_line[4]) + upper

            left = int(parsed_line[2])
            upper = int(parsed_line[3])
            right = int(parsed_line[4]) + left
            lower = int(parsed_line[5]) + upper
       
            if right >= img.size[0]:
                right = img.size[0]-1
            if lower >= img.size[1]:
                lower = img.size[1]-1
            #crop
            cropped_img = img.crop((left, upper, right, lower))
            #flip
            flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)

            cropped_arr_12 = etc.img2array(cropped_img,etc.img_size_12)
            flipped_arr_12 = etc.img2array(flipped_img,etc.img_size_12)
            cropped_arr_24 = etc.img2array(cropped_img,etc.img_size_24)
            flipped_arr_24 = etc.img2array(flipped_img,etc.img_size_24)
            cropped_arr_48 = etc.img2array(cropped_img,etc.img_size_48)
            flipped_arr_48 = etc.img2array(flipped_img,etc.img_size_48)
        
            #for debugging
            #cropped_img.save(etc.pos_dir + str(i)  + ".jpg")
        
            pos_db_line[0,:etc.dim_12] = cropped_arr_12
            pos_db_line[0,etc.dim_12:etc.dim_12+etc.dim_24] = cropped_arr_24
            pos_db_line[0,etc.dim_12+etc.dim_24:] = cropped_arr_48
        
            pos_db_line[1,:etc.dim_12] = flipped_arr_12
            pos_db_line[1,etc.dim_12:etc.dim_12+etc.dim_24] = flipped_arr_24
            pos_db_line[1,etc.dim_12+etc.dim_24:] = flipped_arr_48
        
            pos_db[i-1] = pos_db_line
    #end for load raw data
    pos_db = [elem for elem in pos_db if type(elem) != int]    
    pos_db = np.vstack(pos_db) 

    #neg image cropping
    nid = 0
    neg_file_list = os.listdir(etc.neg_dir)
    neg_db = [0 for n in xrange(len(neg_file_list))]
    neg_img = [0 for n in xrange(len(neg_file_list))]
    for file in neg_file_list:
        img = Image.open(etc.neg_dir + file)
        if len(np.shape(np.asarray(img))) != 3:
            continue
        neg_img[nid] = img
        print nid+1, "/" , len(neg_file_list), "th neg image cropping..."
        neg_db_line = np.zeros((etc.neg_per_img,etc.dim_12), np.float32)
        for neg_iter in xrange(etc.neg_per_img):
                     
            rad_rand = randint(0,min(img.size[0],img.size[1])-1)
            while(rad_rand <= etc.face_minimum):
                rad_rand = randint(0,min(img.size[0],img.size[1])-1)
            
            x_rand = randint(0, img.size[0] - rad_rand - 1)
            y_rand = randint(0, img.size[1] - rad_rand - 1)
            
            neg_cropped_img = img.crop((x_rand, y_rand, x_rand + rad_rand, y_rand + rad_rand))
            neg_cropped_arr = etc.img2array(neg_cropped_img,etc.img_size_12)
            
            #for debugging
            #neg_cropped_img.save(etc.neg_dir + str(f_num) + "_" + str(r) + ".jpg")
                   
            neg_db_line[neg_iter,:] = neg_cropped_arr
        
        neg_db[nid] = neg_db_line
        nid += 1
   
    neg_db = [elem for elem in neg_db if type(elem) != int]
    neg_db = np.vstack(neg_db)
    neg_img = [elem for elem in neg_img if type(elem) != int]
     
    print "Pos: " + str(np.shape(pos_db)[0])
    print "Neg: " + str(np.shape(neg_db)[0])
    
    return [pos_db, neg_db, neg_img]

def load_db_detect_train_12_net_pos():
   
    print "Loading training db..."

    #annot_dir = etc.db_dir + "aflw/data/"
    annot_dir ="/home/gavinpan/workspace/dataset/aflw/data/"
    #annot_fp = open(annot_dir + "annot", "r")
    annot_fp = open("face_rect.txt", "r")
    raw_data = annot_fp.readlines()
    annot_fp.close()
    #pos image cropping
    print "pos data number={}".format(len(raw_data)-1)
    pos_db = [0 for _ in xrange(len(raw_data)-1)]
    for i,line in enumerate(raw_data):
      	if(not line.startswith('#')):
            if(i%5000==0):
                print i, "/", len(raw_data), "th pos image cropping..."
            parsed_line = line.split()
            imagePath = parsed_line[1].strip()
            imagePath = annot_dir + imagePath
            img = Image.open(imagePath)
            #check if not RGB
            if len(np.shape(np.asarray(img))) != 3 or np.shape(np.asarray(img))[2] != etc.input_channel:
                continue
            
            #pos_db_line = np.zeros((2,etc.dim_12 + etc.dim_24 + etc.dim_48), np.float32)
            pos_db_line = np.zeros((2,etc.dim_12), np.float32)

            #left = int(parsed_line[1])
            #upper = int(parsed_line[2])
            #right = int(parsed_line[3]) + left
            #lower = int(parsed_line[4]) + upper

            left = int(parsed_line[2])
            upper = int(parsed_line[3])
            right = int(parsed_line[4]) + left
            lower = int(parsed_line[5]) + upper
       
            if right >= img.size[0]:
                right = img.size[0]-1
            if lower >= img.size[1]:
                lower = img.size[1]-1
            #crop
            cropped_img = img.crop((left, upper, right, lower))
            #flip
            flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)

            cropped_arr_12 = etc.img2array(cropped_img,etc.img_size_12)
            flipped_arr_12 = etc.img2array(flipped_img,etc.img_size_12)
            '''
            cropped_arr_24 = etc.img2array(cropped_img,etc.img_size_24)
            flipped_arr_24 = etc.img2array(flipped_img,etc.img_size_24)
            cropped_arr_48 = etc.img2array(cropped_img,etc.img_size_48)
            flipped_arr_48 = etc.img2array(flipped_img,etc.img_size_48)
            '''
            #for debugging
            #cropped_img.save(etc.pos_dir + str(i)  + ".jpg")
        
            pos_db_line[0,:etc.dim_12] = cropped_arr_12
            '''
            pos_db_line[0,etc.dim_12:etc.dim_12+etc.dim_24] = cropped_arr_24
            pos_db_line[0,etc.dim_12+etc.dim_24:] = cropped_arr_48
            '''
            pos_db_line[1,:etc.dim_12] = flipped_arr_12
            '''
            pos_db_line[1,etc.dim_12:etc.dim_12+etc.dim_24] = flipped_arr_24
            pos_db_line[1,etc.dim_12+etc.dim_24:] = flipped_arr_48
            '''
            pos_db[i-1] = pos_db_line
    #end for load raw data
    pos_db = [elem for elem in pos_db if type(elem) != int]    
    pos_db = np.vstack(pos_db) 
    print "Pos: " + str(np.shape(pos_db)[0])
    return pos_db
def load_db_detect_train_neg():
    #neg image cropping
    nid = 0
    neg_file_list = os.listdir(etc.neg_dir)
    print "neg data number={}".format(len(neg_file_list))
    neg_db = [0 for n in xrange(len(neg_file_list))]
    neg_img = [0 for n in xrange(len(neg_file_list))]
    for file in neg_file_list:
        img = Image.open(etc.neg_dir + file)
        if len(np.shape(np.asarray(img))) != 3:
            continue
        neg_img[nid] = img
        if(nid+1%2000==0):
            print nid+1, "/" , len(neg_file_list), "th neg image cropping..."
        neg_db_line = np.zeros((etc.neg_per_img,etc.dim_12), np.float32)
        for neg_iter in xrange(etc.neg_per_img):
                     
            rad_rand = randint(0,min(img.size[0],img.size[1])-1)
            while(rad_rand <= etc.face_minimum):
                rad_rand = randint(0,min(img.size[0],img.size[1])-1)
            
            x_rand = randint(0, img.size[0] - rad_rand - 1)
            y_rand = randint(0, img.size[1] - rad_rand - 1)
            
            neg_cropped_img = img.crop((x_rand, y_rand, x_rand + rad_rand, y_rand + rad_rand))
            neg_cropped_arr = etc.img2array(neg_cropped_img,etc.img_size_12)
            
            #for debugging
            #neg_cropped_img.save(etc.neg_dir + str(f_num) + "_" + str(r) + ".jpg")
                   
            neg_db_line[neg_iter,:] = neg_cropped_arr
        
        neg_db[nid] = neg_db_line
        nid += 1
   
    neg_db = [elem for elem in neg_db if type(elem) != int]
    neg_db = np.vstack(neg_db)
    neg_img = [elem for elem in neg_img if type(elem) != int]

    print "Neg: " + str(np.shape(neg_db)[0])
    
    return [neg_db, neg_img]

def load_db_cali_train():
   
    print "Loading training db..."

    #annot_dir = etc.db_dir + "aflw/data/"
    annot_dir ="/home/gavinpan/workspace/dataset/aflw/data/"
    #annot_fp = open(annot_dir + "annot", "r")
    annot_fp = open("face_rect.txt", "r")
    raw_data = annot_fp.readlines()
    #only for memory size
    raw_data = raw_data[:3000]
    annot_fp.close()    

    #pos image cropping
    x_db = [0 for _ in xrange(len(raw_data)-1)]
    for i,line in enumerate(raw_data):
        if(not line.startswith('#')):  
            if(i%5000==0):
                print i, "/", len(raw_data), "th pos image cropping..."
            parsed_line = line.split()
            imagePath = parsed_line[1].strip()
            imagePath = annot_dir + imagePath
            #parsed_line = line.split(',')
            #img = Image.open(etc.pos_dir+parsed_line[0])
            img = Image.open(imagePath)
            #check if not RGB
            if len(np.shape(np.asarray(img))) != 3 or np.shape(np.asarray(img))[2] != etc.input_channel:
                continue
        
            left = int(parsed_line[2])
            upper = int(parsed_line[3])
            right = int(parsed_line[4]) + left
            lower = int(parsed_line[5]) + upper
       
            #width
            if right >= img.size[0]:
                right = img.size[0]-1
            #height
            if lower >= img.size[1]:
                lower = img.size[1]-1
        
            x_db_list = [0 for _ in xrange(etc.cali_patt_num)]

            for si,s in enumerate(etc.cali_scale):
                for xi,x in enumerate(etc.cali_off_x):
                    for yi,y in enumerate(etc.cali_off_y):
                    
                        new_left  = left  - x*float(right-left)/s
                        new_upper = upper - y*float(lower-upper)/s
                        new_right = new_left+float(right-left)/s
                        new_lower = new_upper+float(lower-upper)/s
                    
                        new_left  = int(new_left)
                        new_upper = int(new_upper)
                        new_right = int(new_right)
                        new_lower = int(new_lower)

                        if new_left < 0 or new_upper < 0 or new_right >= img.size[0] or new_lower >= img.size[1]:
                            continue

                        cropped_img = img.crop((new_left, new_upper, new_right, new_lower))
                        calib_idx = si*len(etc.cali_off_x)*len(etc.cali_off_y)+xi*len(etc.cali_off_y)+yi

                        #for debugging
                        #cropped_img.save(etc.pos_dir + str(i)  + ".jpg")

                        x_db_list[calib_idx] = [cropped_img,calib_idx]

                        
            x_db_list = [elem for elem in x_db_list if type(elem) != int]
            if len(x_db_list) > 0:
                x_db[i-1] = x_db_list
    #end for loop
    x_db = [elem for elem in x_db if type(elem) != int]    
    x_db = [x_db[i][j] for i in xrange(len(x_db)) for j in xrange(len(x_db[i]))]
    
    return x_db


def load_db_test():
    
    print "Loading test db..."
    
    load_db_start = time.time()

    annot_dir = etc.db_dir + "FDDB-folds/"
    test_img = []
    test_annot = []
    test_face_num = []
       
    
    for f_num in xrange(1,10+1):
        
        print f_num, "/", 10, "folds is loading..."
        
        index = str(f_num).zfill(2)
        annot_file = annot_dir + "FDDB-fold-" + index + "-ellipseList.txt"
        
        fp = open(annot_file)
        raw_data = fp.readlines()
        
        stage = 0
        for parsed_data in raw_data:
                                
            if stage == 0:
                file_name = parsed_data.rstrip()
                stage = 1
            elif stage == 1:
                face_num = int(parsed_data)
                
                test_img.append(file_name)
                test_face_num.append(face_num)
                
                test_annot_line = [0 for r in xrange(face_num)]
                stage = 2

            elif stage == 2:
                splitted = parsed_data.split()
                
                y_half = max([float(splitted[0]) * math.cos(abs(float(splitted[2]))), float(splitted[0]) * math.sin(abs(float(splitted[2])))])
                x_half = max([float(splitted[1]) * math.sin(abs(float(splitted[2]))), float(splitted[1]) * math.cos(abs(float(splitted[2])))])
                
                left = float(splitted[3]) - x_half
                right = float(splitted[3]) + x_half
                upper = float(splitted[4]) - y_half
                lower = float(splitted[4]) + y_half
                
                """
                if x_half < etc.face_minimum/2 or y_half < etc.face_minimum/2:
                    test_face_num[-1] -=1
                    face_num -= 1
                    if face_num == 0:
                        test_annot_line = [elem for elem in test_annot_line if type(elem) != int]
                        test_annot.append(test_annot_line)
                        stage = 0
                    continue
                """

                test_annot_line[test_face_num[-1] - face_num] = [left, upper, right, lower]


                face_num -= 1
                if face_num == 0:
                    test_annot_line = [elem for elem in test_annot_line if type(elem) != int]
                    test_annot.append(test_annot_line)
                    stage = 0

        fp.close()
    
    print "Test: " + str(len(test_img))
    
    load_db_finish = time.time()
    print load_db_finish - load_db_start, "secs for loading db..."
    
    return [test_img, test_annot, test_face_num]



