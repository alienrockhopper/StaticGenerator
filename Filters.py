from functools import partial
from Data import ChannelData, ImageData
import numpy as np
import cv2 as cv

class FilterFactory(object):
  def __init__(self, seed):
    self.random = np.random.default_rng(seed)

  def buildFilter(self, filterConfig, imgData: ImageData):
    if filterConfig['name'] == 'edge':
      return partial(edge, filterConfig['thresholds'])
    if filterConfig['name'] == 'grey':
      return partial(grey)
    if filterConfig['name'] == 'minmax':
      return partial(minmax)
    if filterConfig['name'] == 'colourmask':
      return partial(colourMask, filterConfig['mask'])
    raise Exception("Filter not defined.  Filter Name: %s" % filterConfig['name'])

def edge(thresholds, channel: ChannelData):
  channel.img = cv.cvtColor(cv.Canny(channel.img, thresholds[0], thresholds[1]), cv.COLOR_GRAY2RGB)

def minmax(channel: ChannelData):
  __applyFunc1d(lambda x: np.amax(x) - np.amin(x), channel)
                               
def grey(channel: ChannelData):
  __applyFunc1d(lambda x: np.dot(x, [0.3, 0.59, 0.11]), channel)

def colourMask(mask, channel: ChannelData):
  __applyFunc(lambda x: (x * mask).astype(np.uint8), channel)
  

def __applyFunc1d(foo, channel):
  channel.img = np.apply_along_axis(lambda x: np.repeat(foo(x), 3).astype(np.uint8), 2, channel.img)

def __applyFunc(foo, channel):
  channel.img = np.apply_along_axis(lambda x: foo(x), 2, channel.img)
  
  