
import os
import pathlib
import sys
import argparse
import time
import imageio.v3 as iio
import numpy as np
import cv2 as cv
import json
from Operations import OpFactory
from Filters import FilterFactory
from Data import ChannelData, ImageData, PixelData

def blendChannels(dest, channels: list[ChannelData], mode):
  print("Blending Channels")
  base = channels[0].img
  leny = len(base)
  tw = 0
  for channel in channels:
    tw += channel.weight
  
  with np.nditer(base, flags=['multi_index'], op_flags=['readonly']) as it:
    for item in it:
      r = item
      it.iternext()
      g = it.value
      it.iternext()
      b = it.value
      
      y = it.multi_index[0]
      x = it.multi_index[1]
      
      print(f'Blending x:{x:5} y:{y:5} / {leny:5}', end='\r')
      colour = [0, 0, 0]
      ltw = tw

      if mode == 'bright':
        for channel in channels:
          if channel.img[y][x][0] == 0 and channel.img[y][x][1] == 0 and channel.img[y][x][2] == 0:
            ltw -= channel.weight

      for channel in channels:
        match mode:
          case 'add':
            colour += channel.img[y][x] 
          case 'weighted':
            colour += channel.img[y][x] * ((channel.weight / tw) if ltw > 0 else 1)
          case 'bright':
            colour += channel.img[y][x] * ((channel.weight / ltw) if ltw > 0 else 0)

      dest[y][x] = colour
      
  print("\nFinished Blends")

def processFilters(channels: list[ChannelData]):
  print("Processing Filters")
  for channel in channels:
    print("Processing Channel: ", channel.name)
    if len(channel.filters) != 0:
      for f in channel.filters:
        f(channel)
    print("\nFinished Channel")
  
def processOperations(channels: list[ChannelData]):
  print("Processing Ops")
  for channel in channels:
    print("Processing Channel: ", channel.name)
    if len(channel.ops) == 0:
      continue
    
    im = channel.img.copy()
    channel.img = np.zeros_like(im)
    leny = len(im)
    lenx = len(im[0])
    
    with np.nditer(im, flags=['multi_index'], op_flags=['readonly']) as it:
      for item in it:
        r = item
        it.iternext()
        g = it.value
        it.iternext()
        b = it.value
        
        y = it.multi_index[0]
        x = it.multi_index[1]
        
        print(f'Opsing x:{x:5} y:{y:5} / {leny:5}', end='\r')
        data = PixelData(x, y, [r, g, b])
        for func in channel.ops:
          func(data)
        if (data.y < leny and data.x < lenx):
          channel.img[data.y][data.x] = data.colour
    print("\nFinished Channel")
    
  
def processConfig(config, imageData: ImageData) -> list[ChannelData]:
  src = iio.imread(config['image'])
  out = pathlib.Path(config['out'])
  outDir = out.parent
  if (not outDir.exists()):
    outDir.mkdir()
  
  seed = config['seed'] if 'seed' in config else int(time.time())
  opf = OpFactory(seed)
  ff = FilterFactory(seed)
  channels = []
  
  for channel in config['channels']:
    w = channel['weight'] if 'weighted' in channel else 1
    ch = ChannelData(src.copy(), channel['name'], w, [], [])
    print("Processing Channel Config: %s" % ch.name)
    if 'ops' in channel:
      for op in channel['ops']:
        ch.ops.append(
          opf.buildOp(op, imageData)
        )
        
    if 'filters' in channel:
      for filter in channel['filters']:
        ch.filters.append(
          ff.buildFilter(filter, imageData)
        )
        
    channels.append(ch)
    print("Finished Channel Config: %s" % ch.name)

  return channels

def __appendToFilename(filename, str):
  p = pathlib.Path(filename)
  return "{0}_{1}{2}".format(pathlib.Path.joinpath(p.parent, p.stem), str, p.suffix)
  
def main():
  parser = argparse.ArgumentParser(
                    prog='StaticGen',
                    description='Applies static distortions to an image.')
  
  parser.add_argument('filename', help='The configuration file to use')
  args = parser.parse_args()

  
  with open(args.filename, mode="r", encoding="utf-8") as stream:
    config = json.load(stream)
    
  img = iio.imread(config['image'])
  imageData = ImageData(img.shape[1], img.shape[0])
  
  channels = processConfig(config, imageData)
  processFilters(channels)
  processOperations(channels)
  
  totalWeight = 0.
  for channel in channels:
    totalWeight += channel.weight
    if 'writeChannels' in config and config['writeChannels']:
      iio.imwrite(__appendToFilename(config['out'], f"ch_{channel.name}"), channel.img)
    
  '''
  blended = np.zeros((imageData.height, imageData.width, 3), np.uint8)
  blended1 = np.zeros((imageData.height, imageData.width, 3), np.uint8)
  blended2 = np.zeros((imageData.height, imageData.width, 3), np.uint8)
  
  for channel in channels:
    alpha = channel.weight / totalWeight
    cv.addWeighted(blended, 1-alpha, channel.img, alpha, 0, blended)
  iio.imwrite("./out1/final.png", blended)
  
  cv.addWeighted(channels[0].img, 0.4, channels[1].img, 0.2, 0, blended1)
  cv.addWeighted(channels[2].img, 0.2, channels[3].img, 0.2, 0, blended2)
  
  iio.imwrite("./out1/final1.png", blended1)
  iio.imwrite("./out1/final2.png", blended2)
  '''
  
  blendModes = config['blendMode'] if 'blendMode' in config else 'add'
  for blend in blendModes.split('|'):
    cb = np.zeros((imageData.height, imageData.width, 3), np.uint8)
    blendChannels(cb, channels, blend)
    name = __appendToFilename(config['out'], f"b_{blend}") if '|' in blendModes else config['out']
    iio.imwrite(name, cb)
  
  
    
  return
  if (not os.path.exists("./out/")):
    os.mkdir("./out/")
  # img [col][row]
  img = iio.imread("./img/Earth.png")
  
  img_r = img.copy()
  img_r[:, :, 1] = 0
  img_r[:, :, 2] = 0
  
  img_g = img.copy()
  img_g[:, :, 0] = 0
  img_g[:, :, 2] = 0
  
  img_b = img.copy()
  img_b[:, :, 0] = 0
  img_b[:, :, 1] = 0
  
  #img_roll = np.roll(img.copy(), 100*3)
  
  #iio.imwrite("./out/Earth_roll.png", img_roll)
  #blended = cv.addWeighted(img_roll, 0.5, img, 0.5, 0)
  #iio.imwrite("./out/Earth_blended.png", blended)
  
  rng = np.random.default_rng(12345)
  
  img_moved = np.empty_like(img)
  print ("StartMove")
  process(img,
    [
      [
        img_moved,
        partial(maskColor, (0, 1, 0)),
        partial(shift, (100, 0), (rng.integers(0, 60, img_moved.shape[0]), rng.integers(0, 1, img_moved.shape[1])) ),
        partial(snow, (5, 5))
      ]
    ])
  print ("EndMove")
  iio.imwrite("./out/Earth_moved.png", img_moved)
  img_moved = cv.addWeighted(img_moved, 0.3, img, 0.7, 0)
  iio.imwrite("./out/Earth_blended.png", img_moved)
  print ("WrittenMove")
  
  
  iio.imwrite("./out/Earth_red.png", img_r)
  iio.imwrite("./out/Earth_green.png", img_g)
  iio.imwrite("./out/Earth_blue.png", img_b)
  
  print("Static Done!")

if __name__ == '__main__':
  sys.exit(main())