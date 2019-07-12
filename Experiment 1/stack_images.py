# from PIL import Image
# import os 


# images = map(Image.open, ['test1.jpg', 'test2.png'])
# widths, heights = zip(*(i.size for i in images))

# total_width = sum(widths)
# max_height = max(heights)

# new_im = Image.new('RGB', (total_width, max_height))

# images = map(Image.open, ['test1.jpg', 'test2.png'])

# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]

# new_im.save('yo3.png')

# im = Image.open('test1.jpg')

# img = Image.open('test2.png')

# im.paste(img, (500,0))

# im.save('yo2.png')

from PIL import Image
import os


out_path = '/home/videsh/Downloads/Chandan/data/for_pix2pix/stacked_AB'

def mask_HE( path1 , path2 ):
	i = 0
	j = 0
	for root,_, m in os.walk(path1):
		m = sorted(m)
		for i in range(len(m)):
			print(i)
			mk = Image.open( os.path.join(path1,m[i]))

			for root2,_,He in os.walk(path2):
				He = sorted(He)
				for j in range(len(He)):
					#if 'fake_B' in He[j]:
						#continue
					print(j)
					if i == j:

						he = Image.open(os.path.join(path2,He[j]))
						imgs = [mk ,  he]
						widths,heights = zip(*(i.size for i in imgs))
						total_width =sum(widths)
						max_height =max(heights)
						x_offset = 0
						new_in = Image.new('RGB', (total_width, max_height))
						for im in imgs:
							new_in.paste(im,(x_offset,0))
							x_offset += im.size[0]

			new_in.save(os.path.join(out_path,'{}.png'.format(i)))

mask_HE( '/home/videsh/Downloads/Chandan/data/for_pix2pix/A', '/home/videsh/Downloads/Chandan/data/for_pix2pix/B')









