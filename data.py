"""
i/o and basic data prep tools for clog loss challenge
Jason Manley, jmanley@rockefeller.edu
"""

import os, glob
import urllib.request, tarfile
from tqdm import tqdm
import numpy as np

class ClogData:
	"""
	Prepares train/test data and holds metadata for clog loss challenge
	For more information on data, see https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/217/
	Download links retrieved from https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/data/
	"""
	
	def __init__(self, path, size='nano'):
		self.base_path = path
		self.size = size
		self.path = os.path.join(path, size)

		for url in ['https://s3.amazonaws.com/drivendata-prod/data/65/public/train_metadata.csv',
		            'https://s3.amazonaws.com/drivendata-prod/data/65/public/train_labels.csv',
		            'https://s3.amazonaws.com/drivendata-prod/data/65/public/test_metadata.csv']:
			download(self.path, url)

		if size == 'nano' or size =='micro':
			if not os.path.exists(self.path):
				print('DOWNLOADING AND EXTRACTING', size, 'DATA...')

				os.mkdir(self.path)
				tar_file = os.path.join(self.path, size + '_vessels.tar')

				urllib.request.urlretrieve('https://s3.amazonaws.com/drivendata-public-assets/' + size + '_vessels.tar',
					                       tar_file)

				tar = tarfile.open(tar_file)
				tar.extractall(path=self.base_path)
				tar.close()

				os.remove(tar_file)

			self.vids = glob.glob(os.path.join(self.path, '*.mp4'))

		elif size == 'full':
			pass

		else:
			print("ERROR: size must be 'nano', 'micro', or 'full'")
			raise

	def crop_videos(self, out_size=(150,150)):
		print('CROPPING ALL VIDEOS TO SIZE ', out_size, '...')

		for vid in tqdm(self.vids):
			noext = os.path.splitext(vid)[0]

			if noext[-7:] != 'cropped':
				vidcrop = noext + '_cropped.mp4'
				if not os.path.exists(vidcrop):

					import cv2
					from skimage.transform import resize

					cap = cv2.VideoCapture(vid)

					ret, img = cap.read()
					bbox = detect_ROI(img)

					fourcc = cv2.VideoWriter_fourcc(*'mp4v')
					hz = 10 # frame rate
					vw = cv2.VideoWriter(vidcrop, fourcc, hz, out_size)

					while ret:
						vw.write((resize(crop(img, bbox), out_size, anti_aliasing=True)*255).astype(np.uint8))

						ret, img = cap.read()

		self.vids = glob.glob(os.path.join(self.path, '*_cropped.mp4'))


def download(path, url):
	""" Downloads a file from url if it does not exist in path."""

	out = os.path.join(path, os.path.basename(url))

	if not os.path.exists(out):
		urllib.request.urlretrieve(url, out)


def crop(img, bbox): return img[bbox[0]:bbox[2], bbox[1]:bbox[3],:]


def detect_ROI(img, threshold=[[9,98],[13,143],[104,255]], display=False):
	"""
	Detects circled ROI in img and returns bounding box
	Threshold taken from sample code: https://github.com/drivendataorg/clog-loss-stall-catchers-benchmark/blob/e2f847d81a901d005c7f8bf75093476052c523fd/BenchmarkCodeOnly.m#L211
	"""

	mask = np.ones(img.shape[0:2])

	for i in range(len(threshold)):
	    mask = np.all([mask, img[:,:,i] >= threshold[i][0], img[:,:,i] <= threshold[i][1]], axis=0)

	from skimage.measure import regionprops

	props = regionprops(mask.astype(int))
	prop = props[np.argmax([p.area for p in props])]
	bbox = prop.bbox

	if display:
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle

		plt.imshow(img)
		plt.gca().add_patch(Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0],
			  						  linewidth=1,edgecolor='r',facecolor='none'))

	return bbox