# i = 0
# j= 0

# for i in range(0,1024,256):

# 	if i == 768:

# 		i = 745

# 	for j in range( 0, 1024, 256):
		
# 		if j == 768:

# 			j = 745

# 			print( '{} {}'.format(i,j))
# 			break 

# 		print( '{} {}'.format( i, j))

 
# 	if i == 745:
# 		break

# import random
# k = 0 
# l = 0

# for i in range(4):

# 	for j in range(4):

# 		print( random.randint(0,255)) 

# ---------------------------------------
# one folder image to other
# ---------------------------------------

# from PIL import Image
# import os

# out_dir = '/home/siddharth/Documents/MeDAL_INTERN/new_rs_20'

# def c_i_f(path):
# 	for root,subdirs,files in os.walk(path):
# 		for item in files:

# 			im = Image.open(os.path.join(root,item))
# 			im.save(os.path.join(out_dir,item))

# c_i_f('/home/siddharth/Documents/MeDAL_INTERN/rs_20')


#-------------------------------------------------------------------------------
# final
#---------------------------------------------------------------------------------

from PIL import Image
import os 

import sys
sys.excepthook = sys.__excepthook__


out_dir = '/home/videsh/Downloads/Chandan/NucleiSegmentation-master/train2/'

def c_i_f(path):
	for root,subdirs,files in os.walk(path):
		print(len(files))
		c=195
		files = sorted(files)
		for item in files:
			
			img = Image.open(os.path.join(root,item))
			h, w=img.size
			for i in range(0,h,256):
				print('i ', i)
				if i >= h-256:

					i = h-256-1

				for j in range( 0, w, 256):
					c+=1
					print('j ', j)
					if j == w-256:

						j = w-256-1

						area = ( i, j, i+256, j+256)

						path = img.crop(area)

						path.save(os.path.join(out_dir, f'{c}.png'))

						break

					area = ( i, j, i+256, j+256)

					path = img.crop(area)

					path.save( os.path.join(out_dir, f'{c}.png')) 
 
				if i == h-256-1:
					break



c_i_f('/home/videsh/Downloads/Chandan/NucleiSegmentation-master/train2/')

#----------------------------------------------------------------------------------------------
#Done
#-----------------------------------------------------------------------------------------------




# from PIL import Image


# img = Image.open('TCGA-18-5592-01Z-00-DX1.jpg')

# import sys
# sys.excepthook = sys.__excepthook__


# for i in range(0,1024,256):

# 	if i == 768:

# 		i = 745

# 	for j in range( 0, 1024, 256):
		
# 		if j == 768:

# 			j = 745

# 			area = ( i, j, i+255, j+255)

# 			path = img.crop(area)

# 			path.save('out{}{}.png'.format(i,j))

# 			break

# 		area = ( i, j, i+255, j+255)

# 		path = img.crop(area)

# 		path.save('out{}{}.png'.format(i,j))  

# 		#print( '{} {}'.format( i, j))

 
# 	if i == 745:
# 		break

# import random
# i = 0
# k = 0 

# for i in range(4):
   
    
# 	k = random.randint(0,745)
	
# 	area = ( k, k, k+255, k+255)

# 	path = img.crop(area)

# 	path.save('out{}.png'.format(i))

# 	#print( random.randint(0,745))


   
	
# # 	k = random.randint(0,745)
	
# # 	area = ( k, k, k+255, k+255)

# # 	path = img.crop(area)

# # 	path.save('out{}.png'.format(i))

	







		 




	



	
