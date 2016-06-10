import os
import time

#pos :0
#neg :1
def gen_train_txt(pos_dir,neg_dir,fname):
    fp = open(fname, "w")
    for root,dirs,files in os.walk(pos_dir):
       #print root
       for file in files:
          path1=os.path.join(root,file)
          #print path1
          path2=path1+" 0\n"
          #print path2
          #time.sleep(1000)
          fp.write(path2)
    for root,dirs,files in os.walk(neg_dir):
       #print root
       for file in files:
          path1=os.path.join(root,file)
          #print path1
          path2=path1+" 1\n"
          #print path2
          fp.write(path2)
    fp.close()

def gen_train_cali_txt(pos_dir,fname):
    fp = open(fname, "w")
    for root,dirs,files in os.walk(pos_dir):
       #print root
       #print dirs
       for dir in dirs:
           path1=os.path.join(root,dir)
           for root1,dirs1,files1 in os.walk(path1):
               #print root1
               #print files1
               for file in files1:
                   path2=os.path.join(root1,file)
                   #print path2
                   path3=path2+" "+str(dir)+"\n"
                   #print path3
                   fp.write(path3)
                   #time.sleep(1000)

    fp.close()  

if __name__ == '__main__':

   '''

   train_pos_12_net_dir="net_data/pos_train/"
   train_neg_12_net_dir="net_data/neg_train/"
   train_fname="train_net.txt"
   gen_train_txt(train_pos_12_net_dir,train_neg_12_net_dir,train_fname)

   val_pos_12_net_dir="net_data/pos_val/"
   val_neg_12_net_dir="net_data/neg_val/"
   val_fname="val_net.txt"
   gen_train_txt(val_pos_12_net_dir,val_neg_12_net_dir,val_fname)
   '''

   '''
   train_cali_pos_12_net_dir="cali_net_data/pos_train/"
   train_fname="train_cali_net.txt"
   gen_train_cali_txt(train_cali_pos_12_net_dir,train_fname)
   
   '''
   val_cali_pos_12_net_dir="cali_net_data/pos_val/"
   val_fname="val_cali_net.txt"
   gen_train_cali_txt(val_cali_pos_12_net_dir,val_fname)


