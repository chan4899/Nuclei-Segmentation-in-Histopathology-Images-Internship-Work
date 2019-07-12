from PIL import Image, ImageEnhance
import glob
import os
inp_dir = "/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/dataset/DRIVE/training/hne_448/"
lis = glob.glob(inp_dir + "*.png")
print(len(lis))
c=0
out_dir = '/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/dataset/DRIVE/training/contrast_hne_448/'
for im in lis: 
	c = c+1
	print(c, im)
	name = im.split('/')[-1].split('.')[0]
	im = Image.open(im)
	enhancer = ImageEnhance.Contrast(im)
	enhanced_im = enhancer.enhance(4.0)
	# nam = name.split('.')[0]
	enhanced_im.save(out_dir + name + '.png')
