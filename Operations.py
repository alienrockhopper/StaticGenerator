from functools import partial
from Data import PixelData, ImageData
import numpy as np
import random as rnd

class OpFactory(object):
  def __init__(self, seed):
    self.random = np.random.default_rng(seed)
  
  def buildOp(self, opConfig, imgData: ImageData):
    if opConfig['name'] == 'snow':
      return partial(snow, opConfig['delta'])
    elif opConfig['name'] == "shift":
      delta =  opConfig['delta'] if 'delta' in opConfig else [0,0]
      rndDelta = opConfig['randomDelta'] if 'randomDelta' in opConfig else [0,0]
        
      return partial(shift, delta, (
        self.random.integers(min(0, rndDelta[0]), max(1, rndDelta[0]+1), imgData.height), 
        self.random.integers(min(0, rndDelta[1]), max(1, rndDelta[1]+1), imgData.width)) 
      )
    elif opConfig['name'] == "colourmask":
      return partial(maskColour, opConfig['mask'])
    
    raise Exception("Operation not defined.  Operation Name: %s" % opConfig['name'])

def snow(random, data: PixelData):
  data.x += rnd.randint(0, random[0]) 
  data.y += rnd.randint(0, random[1])

def shift(shift, random, data: PixelData):
  x = data.x
  data.x += shift[0] + random[0][data.y]
  data.y += shift[1] + random[1][x]

def maskColour(mask, data: PixelData):
  data.colour[0] = data.colour[0] * mask[0]
  data.colour[1] = data.colour[1] * mask[1]
  data.colour[2] = data.colour[2] * mask[2]