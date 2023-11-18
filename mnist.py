# CS4412 : Data Mining
# Kennesaw State University

"""Basic structure for the MNIST handwritten digits dataset, for use
in this project.

"""

import numpy as np                    # matrix library
from matplotlib import pyplot         # for plotting
from mpl_toolkits.axes_grid1 import ImageGrid # for drawing images
import pickle                         # reading/writing python objects

class Mnist:
    def __init__(self,mnist_file="train",mnist_dir="mnist"):
        self._dim = (28,28) # dimension of images (28x28)
        # read in the dataset
        with open("%s/mnist-%s-labels" % (mnist_dir,mnist_file),"rb") as f:
            self._labels = pickle.load(f)
        with open("%s/mnist-binary-%s-images" % (mnist_dir,mnist_file),"rb") as f:
            self._images = pickle.load(f)
        self._N = len(self._labels) # size of dataset

    def images(self):
        return self._images

    def labels(self):
        return self._labels

    def image_labels_of_digits(self,digits):
        indices = np.zeros(self._N).astype('bool')
        for digit in digits:
            digit_indices = self._labels == digit
            indices = np.logical_or(indices,digit_indices)
        images = self._images[indices]
        labels = self._labels[indices]
        return images,labels

    def shrink(self,factor=0.5):
        """Shrinks the dataset by a factor (default .5).
        This is used to speed things up for testing."""
        self._N = int(factor*self._N)
        self._labels = self._labels[:self._N]
        self._images = self._images[:self._N]

    def save_ten_as_image(self,ten,filename,title=""):
        """Input ten images to save in one"""
        assert len(ten) == 10
        fig = pyplot.figure(figsize=(10,5))
        grid = ImageGrid(fig,111,nrows_ncols=(2,5),axes_pad=0.1)
        for ax,im in zip(grid,ten):
            im = im.reshape(self._dim)
            ax.imshow(im,cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title)
        pyplot.savefig(filename)

    def save_image(self,image,filename,cmap="gray",limit=None,title=None):
        fig = pyplot.figure()
        image = image.reshape(self._dim)
        if limit:
            pyplot.imshow(image,cmap=cmap,vmin=-limit,vmax=limit)
        else:
            pyplot.imshow(image,cmap=cmap)
        if title:
            pyplot.title(title)
        pyplot.savefig(filename)
        pyplot.close()

    def plot_image(self,image,second=None,cmap="gray",limit=None):
        fig = pyplot.figure()
        image = image.reshape(self._dim)
        if limit:
            pyplot.imshow(image,cmap=cmap,vmin=-limit,vmax=limit)
        else:
            pyplot.imshow(image,cmap=cmap)
        if second is not None:
            second = second.reshape(self._dim)
            pyplot.imshow(second,cmap="gray",alpha=0.5)
        pyplot.show()
        pyplot.close()
