from PIL import Image
import PIL.ImageOps
import glob

im = glob.glob("/home/videsh/Downloads/Chandan/NucleiSegmentation-master/train/trainB/*.png") 

print(len(im))   
for i in im:
	image = Image.open(i)
	print("ff")
	inverted_image = PIL.ImageOps.invert(image)
	j=i.replace("NucleiSegmentation-master/train/trainB", "fake_B_inverted")
	inverted_image.save(j)
