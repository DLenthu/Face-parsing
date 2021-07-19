# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import os
import cv2
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    
# base_dir = "/home/bhargav/Documents/GAN"

 
# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = cv2.resize(cv2.imread(filename),size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

if __name__ == "__main__": 
# load source image
  src_image = load_image('test.png')
  print('Loaded', src_image.shape)
  # load model
  model = load_model('pix2pix_700.h5')
  # generate image from source
  gen_image = model.predict(src_image)
  # scale from [-1,1] to [0,1]
  gen_image = (gen_image + 1) / 2.0
  # plot the image
  pyplot.imshow(gen_image[0])
  pyplot.axis('off')
  pyplot.show()