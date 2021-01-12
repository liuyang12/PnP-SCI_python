import os

res = '480p'
source_dir = '/data/yliu/proj/dataset/video/DAVIS/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/' + res
result_dir = '/data/yliu/proj/dataset/video/DAVIS/DAVIS-2017-trainval-480p/DAVIS/video/' + res



filenames= os.listdir (source_dir) # get all files' and folders' names in the current directory

result = []
for filename in filenames: # loop through all the files and folders
    if os.path.isdir(os.path.join(os.path.abspath(source_dir), filename)): # check whether the current object is a folder or not
        result.append(filename)

os.makedirs(result_dir)
for index,filename in enumerate(result):
    exeline = 'ffmpeg -hide_banner -loglevel panic -i %s/%s/%%05d.jpg %s/%s_%s.mp4' % (source_dir,filename,result_dir,res,filename)
    os.system(exeline)


