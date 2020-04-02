import argparse
import glob
import json
import logging
import multiprocessing
import numpy as np
import os
import pprint
import random
import subprocess
import sys
import tempfile

from laspy.file import File as LasFile
from laspy.header import Header as LasHeader
from pathlib import Path

# Point set (point cloud) class definition
# input is file...
class PointSet(object):
    def __init__(self, points_file, class_file=''):
        # Load points file
        if points_file.endswith('.las') or points_file.endswith('.laz'):
            lfile = LasFile(points_file, mode='r')
            self.x = np.copy(lfile.X).astype('f8')*lfile.header.scale[0]
            self.y = np.copy(lfile.Y).astype('f8')*lfile.header.scale[1]
            self.z = np.copy(lfile.Z).astype('f8')*lfile.header.scale[2]
            self.i = np.copy(lfile.Intensity).astype('f8')
            self.r = np.copy(lfile.return_num).astype('f8')
            self.c = np.copy(lfile.Classification)
            lfile.close()
        elif points_file.endswith('.txt'):
            data = np.loadtxt(points_file,delimiter=' ',dtype='f8')
            self.x = data[:,0]
            self.y = data[:,1]
            self.z = data[:,2]
            self.i = data[:,3]
            self.r = data[:,4]
            if not class_file:
                if data.shape[1] > 5:
                    self.c = data[:,5].astype('uint8')
                else:
                    self.c = np.zeros(self.x.shape,dtype='uint8')
            else:
                self.c = np.loadtxt(class_file,dtype='uint8')
        else:
            raise ValueError('Unknown file type extension: '+points_file)
        self.filepath = points_file
        self.filename = os.path.splitext(os.path.basename(points_file))[0]
        if self.filename.endswith('_PC3'):
            self.filename = self.filename[:-4]

    def save(self, output_file, class_file=''):
        if output_file.endswith('.txt'):
            if (not class_file and self.c.any()):
                np.savetxt(output_file,
                        np.stack([self.x,self.y,self.z,self.i,self.r,self.c],axis=1),
                        fmt='%.2f,%.2f,%.2f,%d,%d,%d')
            else:
                np.savetxt(output_file,
                        np.stack([self.x,self.y,self.z,self.i,self.r],axis=1),
                        fmt='%.2f,%.2f,%.2f,%d,%d')
            if class_file:
                self.save_classifications_txt(class_file)
        elif output_file.endswith('.las') or output_file.endswith('.laz'):
            lfile = LasFile(output_file, mode='w', header=LasHeader(x_scale=0.01,y_scale=0.01,z_scale=0.01))
            lfile.X = self.x/0.01
            lfile.Y = self.y/0.01
            lfile.Z = self.z/0.01
            lfile.Intensity = self.i
            lfile.flag_byte = self.r
            lfile.Classification = self.c
            lfile.close()
        else:
            raise ValueError('Unknown file type extension: '+output_file)

    def savedata_txt(self,data,outfile):
        np.savetxt(outfile,data,fmt='%.2f,%.2f,%.2f,%d,%d')
    def savelabel_txt(self,label,outfile):
        np.savetxt(outfile,label, fmt='%d')

    
    def save_canonical_txt(self, output_file_name_base):
        self.save(output_file_name_base+'_PC3.txt',
                class_file=output_file_name_base+'_CLS.txt')
                
    def save_classifications_txt(self, class_file):
        np.savetxt(class_file,self.c,fmt='%d')

    def cloud2blocks_test(self,block_size=100,stride=100):
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        ymin = np.min(self.y)
        ymax = np.max(self.y)
        xbeg_list= []
        ybeg_list= []
        num_block_x = int(np.around((xmax-xmin-block_size) / stride))+1
        num_block_y = int(np.around((ymax-ymin-block_size) / stride))+1  # note this t_cox is 405
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(xmin+i*stride)
                ybeg_list.append(ymin+j*stride)

        block_data_list=[]
        block_label_list=[]
        for idx in range(len(xbeg_list)): # idx=0...15
            xbeg=xbeg_list[idx]
            ybeg=ybeg_list[idx]
            if idx==3 or idx==7 or idx==11 or idx==15:
                xcond=(self.x<=xbeg+block_size) & (self.x>=xbeg)
                ycond=(self.y<=ymax) & (self.y>=ybeg)
            else:
                xcond=(self.x<=xbeg+block_size) & (self.x>=xbeg)
                ycond=(self.y<=ybeg+block_size) & (self.y>=ybeg)
            cond=xcond&ycond
            if np.sum(cond)<100: # discard block if there are less than 100 pts
                continue
            # np.stack([self.x, self.y, self.z, self.i, self.r, self.c], axis=1)
            block_data=np.stack([self.x[cond],self.y[cond],self.z[cond],self.i[cond],self.r[cond]],axis=1)
            block_label=self.c[cond]
            block_data_list.append(block_data)
            block_label_list.append(block_label)

        return block_data_list,block_label_list

    def cloud2blocks_finaltest(self,block_size=25,stride=12.5):
        [block_data_list,block_label_list]=self.cloud2blocks_test()
        block_data_total=[]
        block_label_total=[]

        for k in range(block_data_list.__len__()):  # 0....12
            # l=block_data_list[k].__len__()
            xmin=np.min(block_data_list[k][:,0])
            xmax=np.max(block_data_list[k][:,0])
            ymin=np.min(block_data_list[k][:,1])
            ymax=np.max(block_data_list[k][:,1])
            xbeg_list = []
            ybeg_list = []
            num_block_x = int(np.ceil((xmax - xmin - block_size) / stride)) + 1
            num_block_y = int(np.ceil((ymax - ymin - block_size) / stride)) + 1
            for i in range(num_block_x):
                for j in range(num_block_y):
                    xbeg_list.append(xmin + i * stride)
                    ybeg_list.append(ymin + j * stride)
            # collect blokcs for each 100*100 block
            block_data_one=[]
            block_label_one=[]

            for idx in range(len(xbeg_list)):  # idx=0......
                xbeg = xbeg_list[idx]
                ybeg = ybeg_list[idx]
                xcond = (block_data_list[k][:,0] <= xbeg + block_size) & (block_data_list[k][:,0] >= xbeg)
                ycond = (block_data_list[k][:,1]<= ybeg + block_size) &  (block_data_list[k][:,1] >= ybeg)
                cond = xcond & ycond
                if np.sum(cond) < 10:  # discard block if there are less than 10 pts
                    continue

                block_data = block_data_list[k][cond,:]
                block_label = block_label_list[k][cond]
                block_data_one.append(block_data)
                block_label_one.append(block_label)
            block_data_total.append(block_data_one)  # maybe concat
            block_label_total.append(block_label_one)
        return block_data_total,block_label_total

    def cloud2blocks(self,block_size=100,stride=100):
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        ymin = np.min(self.y)
        ymax = np.max(self.y)
        xbeg_list= []
        ybeg_list= []
        num_block_x = int(np.around((xmax-xmin-block_size) / stride))+1
        num_block_y = int(np.around((ymax-ymin-block_size) / stride))+1  # note this t_cox is 405
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(xmin+i*stride)
                ybeg_list.append(ymin+j*stride)

        block_data_list=[]
        block_label_list=[]
        for idx in range(len(xbeg_list)): # idx=0...15
            xbeg=xbeg_list[idx]
            ybeg=ybeg_list[idx]
            if idx==3 or idx==7 or idx==11 or idx==15:
                xcond=(self.x<=xbeg+block_size) & (self.x>=xbeg)
                ycond=(self.y<=ymax) & (self.y>=ybeg)
            else:
                xcond=(self.x<=xbeg+block_size) & (self.x>=xbeg)
                ycond=(self.y<=ybeg+block_size) & (self.y>=ybeg)
            cond=xcond&ycond
            if np.sum(cond)<100: # discard block if there are less than 100 pts
                continue
            # np.stack([self.x, self.y, self.z, self.i, self.r, self.c], axis=1)
            block_data=np.stack([self.x[cond],self.y[cond],self.z[cond],self.i[cond],self.r[cond]],axis=1)
            block_label=self.c[cond]
            block_data_list.append(block_data)
            block_label_list.append(block_label)

        return block_data_list,block_label_list

    def cloud2blocks_train(self,block_size=25,stride=12.5):
        [block_data_list,block_label_list]=self.cloud2blocks()
        block_data_total=[]
        block_label_total=[]

        for k in range(block_data_list.__len__()):  # 0....12
            # l=block_data_list[k].__len__()
            xmin=np.min(block_data_list[k][:,0])
            xmax=np.max(block_data_list[k][:,0])
            ymin=np.min(block_data_list[k][:,1])
            ymax=np.max(block_data_list[k][:,1])
            xbeg_list = []
            ybeg_list = []
            num_block_x = int(np.ceil((xmax - xmin - block_size) / stride)) + 1
            num_block_y = int(np.ceil((ymax - ymin - block_size) / stride)) + 1
            for i in range(num_block_x):
                for j in range(num_block_y):
                    xbeg_list.append(xmin + i * stride)
                    ybeg_list.append(ymin + j * stride)
            # collect blokcs for each 100*100 block
            block_data_one=[]
            block_label_one=[]

            for idx in range(len(xbeg_list)):  # idx=0......
                xbeg = xbeg_list[idx]
                ybeg = ybeg_list[idx]
                xcond = (block_data_list[k][:,0] <= xbeg + block_size) & (block_data_list[k][:,0] >= xbeg)
                ycond = (block_data_list[k][:,1]<= ybeg + block_size) &  (block_data_list[k][:,1] >= ybeg)
                cond = xcond & ycond
                if np.sum(cond) < 100:  # discard block if there are less than 100 pts
                    continue

                block_data = block_data_list[k][cond,:]
                block_label = block_label_list[k][cond]

                block_data_one.append(block_data)
                block_label_one.append(block_label)
            block_data_total.append(block_data_one)  # maybe concat
            block_label_total.append(block_label_one)
        return block_data_total,block_label_total



    def split(self, points_per_chip=4096, overlap=True, pad=0):
        with tempfile.TemporaryDirectory() as tmpdir:
            ipath = os.path.join(tmpdir,self.filename+'.las')
            self.save(ipath)
            
            if overlap:
                charkey = [['A','B'],['C','D']]
                
                # get min/max x/y
                xmin = np.min(self.x)
                xmax = np.max(self.x)
                ymin = np.min(self.y)
                ymax = np.max(self.y)
                
                for i in range(2):
                    xcrop = [xmin+pad,xmax-pad] if i else [xmin-0.5,xmax+0.5]
                    for j in range(2):
                        ycrop = [ymin+pad,ymax-pad] if j else [ymin-0.5,ymax+0.5]
                        
                        opath = os.path.join(tmpdir,
                                self.filename+'_'+charkey[j][i]+'#.las')
                        
                        crop_bounds = '([{},{}],[{},{}])'.format(xcrop[0], xcrop[1], *ycrop)
                        
                        # Format pipeline string
                        if i==0 and j==0:
                            pipeline = {'pipeline':[
                                    ipath,
                                    {'type':'filters.voxelcentroidnearestneighbor','cell':0.4},
                                    {'type':'filters.chipper','capacity':'4096'},
                                    opath]}
                        else:
                            pipeline = {'pipeline':[
                                    ipath,
                                    {'type':'filters.crop','bounds':crop_bounds},
                                    {'type':'filters.voxelcentroidnearestneighbor','cell':0.4},
                                    {'type':'filters.chipper','capacity':str(points_per_chip)},
                                    opath]}
                        
                        p = subprocess.run(['/opt/conda/envs/cpdal-run/bin/pdal','pipeline','-s'],input=json.dumps(pipeline).encode())
                        if p.returncode:
                            raise ValueError('Failed to run pipeline: \n"'+json.dumps(pipeline)+'"')
                        
            else:
                opath = os.path.join(tmpdir,self.filename+'_#.las')
                pipeline = {'pipeline':[
                        ipath,
                        {'type':'filters.voxelcentroidnearestneighbor','cell':0.5},
                        {'type':'filters.chipper','capacity':'8192'},
                        opath]}
                p = subprocess.run(['/opt/conda/envs/cpdal-run/bin/pdal','pipeline','-s'],input=json.dumps(pipeline).encode())
                if p.returncode:
                    raise ValueError('Failed to run pipeline: \n"'+json.dumps(pipeline)+'"')
                
            return [PointSet(str(f)) for f in Path(tmpdir).glob(self.filename+'_*.las')]  # form a class object list : psets


if __name__ == '__main__':
    #main(sys.argv)
    # pset = PointSet('/data/dfc_v4c/test2/classed/OMA/OMA_Tile_095_classes.las')
    pset= PointSet('data/dfc/ISPRSdata/EVAL_PC3.txt','data/dfc/ISPRSdata/EVAL_CLS.txt')  # data/dfc/ISPRSdata/Debug
    # pset = PointSet(pc3_path,cls_path)
    xmin = np.min(pset.x)
    xmax = np.max(pset.x)
    ymin = np.min(pset.y)
    ymax = np.max(pset.y)
    size=pset.x.__len__()
    print(xmax)
    print(xmin)
    print(ymax)
    print(ymin)

    print(pset.c[0])
    a=[1,2,3,4,5,6]
    print(a[0:3])
    [x,y]=pset.cloud2blocks_finaltest()

    for i in range(x[2].__len__()):
        out_str1 = pset.filename + '_'+'data2'+'_' + str(i)+'_'+'.txt'
        out_str2 = pset.filename + '_' + 'label2' + '_' + str(i) + '_' + '.txt'
        out_file1 = os.path.join('data/dfc/ISPRSdata/Debug/sub_block_test', out_str1)
        out_file2 = os.path.join('data/dfc/ISPRSdata/Debug/sub_block_test', out_str2)
        data=x[2][i]
        label=y[2][i]
        pset.savedata_txt(data,out_file1)
        pset.savelabel_txt(label,out_file2)

    print(x.__len__())
    print(y.__len__())

    # psets=pset.split()
    # psets = pset.split()
    # print('xmin',xmin)
    # print('xmax', xmax)
    # print('ymin', ymin)
    # print('ymax', ymax)
    # print(psets.__len__())
    # n=psets.__len__()
    # for k in range(n):
    #     filename=psets[k].filename
    #     out_str=filename+'.txt'
    #     out_file=os.path.join('data/dfc/ISPRSdata/Debug/EVAL_1',out_str)
    #     out=open(out_file,'w')
    #     l=len(psets[k].x)
    #     for m in range(l):
    #         out.write('%f %f %f %d %d\n' %(psets[k].x[m],psets[k].y[m],psets[k].z[m],psets[k].i[m],psets[k].r[m]))
    #
    # out.close()
