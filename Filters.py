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
    if filterConfig['name'] == 'erode':
      size = filterConfig['size'] if 'size' in filterConfig else 1
      shape = filterConfig['shape'] if 'shape' in filterConfig else 'rect'
      match shape:
        case 'rect':
          shape = cv.MORPH_RECT
        case 'cross':
          shape = cv.MORPH_CROSS
        case 'ellipse':
          shape = cv.MORPH_ELLIPSE
          
      return partial(erode, size, shape)
    if filterConfig['name'] == 'dilate':
      size = filterConfig['size'] if 'size' in filterConfig else 1
      shape = filterConfig['shape'] if 'shape' in filterConfig else 'rect'
      match shape:
        case 'rect':
          shape = cv.MORPH_RECT
        case 'cross':
          shape = cv.MORPH_CROSS
        case 'ellipse':
          shape = cv.MORPH_ELLIPSE
          
      return partial(dilate, size, shape)
    if filterConfig['name'] == 'saturation':
      scale = filterConfig['scale']
      return partial(saturation, scale)
    if filterConfig['name'] == 'hsvmask':
      scale = filterConfig['mask']
      return partial(hsvMask, scale)
    if filterConfig['name'] == 'interlace':
      size = filterConfig['size'] if 'size' in filterConfig else 5
      lightScale = filterConfig['lightScale'] if 'lightScale' in filterConfig else 0.8
      lerpLines = filterConfig['lerpLines'] if 'lerpLines' in filterConfig else False
      return partial(interlace, size, lightScale, lerpLines)
    
    raise Exception("Filter not defined.  Filter Name: %s" % filterConfig['name'])

def edge(thresholds, channel: ChannelData):
  channel.img = cv.cvtColor(cv.Canny(channel.img, thresholds[0], thresholds[1]), cv.COLOR_GRAY2RGB)

def minmax(channel: ChannelData):
  __applyFunc1d(lambda x: np.amax(x) - np.amin(x), channel)
                               
def grey(channel: ChannelData):
  __applyFunc1d(lambda x: np.dot(x, [0.3, 0.59, 0.11]), channel)

def colourMask(mask, channel: ChannelData):
  __applyFunc(lambda x: (x * mask).astype(np.uint8), channel)

def erode(size, shape, channel: ChannelData):
  element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                      (size, size))
  
  channel.img = cv.erode(channel.img, element)

def dilate(size, shape, channel: ChannelData):
  element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                      (size, size))
  
  channel.img = cv.dilate(channel.img, element)

def saturation(scale, channel: ChannelData):
  channel.img = cv.cvtColor(channel.img, cv.COLOR_RGB2HSV_FULL)
  __applyFunc(lambda x: (np.clip(x * [1, scale, 1], 0, 255)).astype(np.uint8), channel)
  channel.img = cv.cvtColor(channel.img, cv.COLOR_HSV2RGB_FULL).astype(np.uint8)

def hsvMask(mask, channel: ChannelData):
  channel.img = cv.cvtColor(channel.img, cv.COLOR_RGB2HSV_FULL)
  __applyFunc(lambda x: (np.clip(x * mask, 0, 255)).astype(np.uint8), channel)
  channel.img = cv.cvtColor(channel.img, cv.COLOR_HSV2RGB_FULL).astype(np.uint8)

class InterlaceState:
  def __init__(self, size, rowWidth):
    self.rowCount = 0
    self.lineCount = 0
    self.isHigh = True
    self.lineSize = size
    self.width = rowWidth
    
  def factor(self):
    return np.interp(self.lineCount, [0., self.lineSize/2.0, self.lineSize], [1.3, 1., 1.3])
    
def interlace(size, lightScale, lerpLines, channel: ChannelData):
  state = InterlaceState(size, channel.img.shape[1])
  def f(state: InterlaceState, x):
    state.rowCount += 1
    if (state.rowCount >= state.width):
      state.rowCount = 0
      state.lineCount += 1
      if (state.lineCount >= state.lineSize):
        state.lineCount = 0
        state.isHigh = not state.isHigh
    
    a = state.factor() if lerpLines else 1
    return np.clip(x * [1, 1, (lightScale * a if not state.isHigh else 1)], 0, 255).astype(np.uint8)
   
  channel.img = cv.cvtColor(channel.img, cv.COLOR_RGB2HSV_FULL)
  __applyFunc(partial(f, state), channel)
  channel.img = cv.cvtColor(channel.img, cv.COLOR_HSV2RGB_FULL).astype(np.uint8)
  

def __applyFunc1d(foo, channel):
  channel.img = np.apply_along_axis(lambda x: np.repeat(foo(x), 3).astype(np.uint8), 2, channel.img)

def __applyFunc(foo, channel):
  channel.img = np.apply_along_axis(lambda x: foo(x), 2, channel.img)
