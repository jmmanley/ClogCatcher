"""
i/o and basic data prep tools for clog loss challenge
Jason Manley, jmanley@rockefeller.edu
"""

import os, glob, sys
import urllib.request, tarfile
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import boto3
import warnings
import dill

class ClogData:
	"""
	Prepares train/test data and holds metadata for clog loss challenge
	For more information on data, see:
	https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/217/
	Download links retrieved from:
	https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/data/

	Size = 'nano', 'micro', or 'full' specifies which pre-defined training set
	size to load (for details, see the DrivenData documentation).
	If toDisk=True, all the movies are automatically downloaded and cropped to
	img_size.
	For 'nano' and 'micro', because the datasets are small, both the full and
	cropped videos are kept.

	METHODS:
	ClogData.crop_videos(vids=self.vids) crops either the videos in self.vids (default)
	or specified input videos.
	ClogData.load(index, train=True, ret=True) loads and returns row index
	of the the training dataset, downloading from the AWS S3 bucket if necessary.
	ClogData.load_train(ret=True) loads and returns the entire training set.
	"""

	def __init__(self, path, size='nano', img_size=(224,224), toDisk=True):
		self.base_path = path
		self.size = size
		self.path = os.path.join(path, size)
		self.img_size = img_size

		# LOAD METADATA
		for url in ['https://s3.amazonaws.com/drivendata-prod/data/65/public/train_metadata.csv',
		            'https://s3.amazonaws.com/drivendata-prod/data/65/public/train_labels.csv',
		            'https://s3.amazonaws.com/drivendata-prod/data/65/public/test_metadata.csv']:
			download(self.base_path, url)

		self.train = pd.read_csv(os.path.join(self.base_path, 'train_metadata.csv'))
		self.train = self.train.merge(pd.read_csv(os.path.join(self.base_path, 'train_labels.csv')))
		self.test  = pd.read_csv(os.path.join(self.base_path, 'test_metadata.csv'))

		if not os.path.exists(os.path.join(self.base_path, 'test')):
			os.mkdir(os.path.join(self.base_path, 'test'))

		# IF NANO/MICRO, DOWNLOAD VIDEOS
		if (size == 'nano' or size =='micro') and toDisk:
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

			self.crop_videos()

			self.train = self.train.groupby(size).get_group(True)

		elif size == 'full':
			if toDisk:
				if not os.path.exists(self.path): os.mkdir(self.path)

				self.vids = glob.glob(os.path.join(self.path, '*_cropped.mp4'))

		else:
			print("ERROR: size must be 'nano', 'micro', or 'full'")
			raise


	def crop_videos(self, force=False, vids=None):
		"""Crops videos in self.vids around ROIs and resize."""
		#TODO: should this be a method of ClogData?
		#      only useful if self.vids is modified by user?

		if vids is None: vids = self.vids

		print('CROPPING ALL VIDEOS TO SIZE ', self.img_size, '...')

		for vid in tqdm(vids, file=sys.stdout):
			noext = os.path.splitext(vid)[0]

			if noext[-7:] != 'cropped':
				vidcrop = noext + '_cropped.mp4'
				if not os.path.exists(vidcrop) or force:

					cap = cv2.VideoCapture(vid)

					ret, img = cap.read()
					bbox = detect_ROI(img)

					fourcc = cv2.VideoWriter_fourcc(*'mp4v')
					hz = 10 # frame rate
					vw = cv2.VideoWriter(vidcrop, fourcc, hz, self.img_size)

					while ret:
						vw.write(crop(img, bbox, self.img_size))

						ret, img = cap.read()

					cap.release()

		vids = glob.glob(os.path.join(self.path, '*_cropped.mp4'))


	def load(self, index, train=True, ret=True, toDisk=True):
		"""
		Loads the video corresponding to row index of train or test set, either
		by retrieving cropped video from disk, or downloading and cropping.
		If ret is True, returns a numpy array of the video.
		If toDisk is True, saves a copy of cropped video to disk.
		"""

		if train: df = self.train
		else: df = self.test

		vid   = df.loc[index].filename
		noext = os.path.splitext(vid)[0]
		if train:
			cropped_vid = os.path.join(self.path, noext + '_cropped.mp4')
		else:
			cropped_vid = os.path.join(self.base_path, 'test', noext + '_cropped.mp4')

		if os.path.exists(cropped_vid):
			# LOAD FROM DISK
			if ret: return load_video(cropped_vid)

		else:
			if not ret and not toDisk: return

			# DOWNLOAD FROM AWS S3 BUCKET AND CROP
			from tempfile import mkdtemp
			tmp = mkdtemp() #temp directory for raw, un-cropped video

			download(tmp, df.loc[index].url)

			vid = load_video(os.path.join(tmp, df.loc[index].filename))
			bbox = detect_ROI(np.squeeze(vid[0,:,:,:]))

			from shutil import rmtree
			rmtree(tmp)

			if ret:
				imgs = np.zeros((vid.shape[0],self.img_size[0],self.img_size[1],3), dtype=np.uint8)

			if toDisk:
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				hz = 10 # frame rate
				vw = cv2.VideoWriter(cropped_vid, fourcc, hz, self.img_size)

			for i in range(vid.shape[0]):
				img = crop(vid[i,:,:,:], bbox, self.img_size)

				if ret: imgs[i,:,:,:] = img
				if toDisk: vw.write(img)

			if ret:
				return imgs


	def load_train(self, ret=True):
		"""Wrapper for ClogData.load() that loads entire training dataset."""

		if self.size != 'full' and ret:
			warnings.warn('Probably a bad idea to load full training set, perhaps use load_train(ret=False) to just download to disk.')

		vids = []

		print('LOADING TRAINING DATA...')
		for i in tqdm(self.train.index):
			vids.append(self.load(i, train=True, ret=ret))

		return vids


	def save(self, out):
		"""Saves ClogData object to file out."""

		dill.dump(self, file = open(out, "wb"))



"""
HELPER FUNCTIONS
"""


def download(path, url):
	"""Downloads a file from url if it does not exist in path."""

	if url[0:2] == 's3':
		s3 = boto3.client('s3')

		pieces = url.split('/')
		bucket = pieces[2]

		s3.download_file(bucket, os.path.join(pieces[3], pieces[4]),
			             os.path.join(path, pieces[4]))

	else:
		out = os.path.join(path, os.path.basename(url))

		if not os.path.exists(out):
			urllib.request.urlretrieve(url, out)


def load_video(vid):
	"""Loads video into NxHxWx3 numpy array."""

	cap = cv2.VideoCapture(vid)
	n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	video = np.zeros((n,h,w,3), np.dtype('uint8'))

	i = 0
	ret = True

	while (i < n  and ret):
	    ret, video[i] = cap.read()
	    i += 1

	cap.release()

	return video


def crop(img, bbox, out_size=None):
	"""Crops HxWx_ image to bbox, resizes if out_size is given."""
	
	from skimage.transform import resize

	img = img[bbox[0]:bbox[2], bbox[1]:bbox[3],:]

	if out_size is not None:
		img = (resize(img, out_size, anti_aliasing=True)*255).astype(np.uint8)

	return img


def detect_ROI(img, threshold=[[9,98],[13,143],[104,255]], display=False):
	"""
	Detects circled ROI in img and returns bounding box
	Threshold taken from sample code:
	https://github.com/drivendataorg/clog-loss-stall-catchers-benchmark/blob/e2f847d81a901d005c7f8bf75093476052c523fd/BenchmarkCodeOnly.m#L211
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
