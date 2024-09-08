
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
          if channel.img[y][x][0] < 10 and channel.img[y][x][1] < 10 and channel.img[y][x][2] < 10:
            ltw -= channel.weight

      for channel in channels:
        match mode:
          case 'add':
            colour = np.clip(colour + channel.img[y][x], 0, 255)
          case 'weighted':
            colour = np.clip(colour + channel.img[y][x] * ((channel.weight / tw) if ltw > 0 else 1))
          case 'bright':
            colour = np.clip(colour + channel.img[y][x] * ((channel.weight / ltw) if ltw > 0 else 0))

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
  
  blendModes = config['blendMode'] if 'blendMode' in config else 'add'
  for blend in blendModes.split('|'):
    cb = np.zeros((imageData.height, imageData.width, 3), np.uint8)
    blendChannels(cb, channels, blend)
    
    if 'postFilters' in config:
      if 'writeChannels' in config and config['writeChannels']:
        name = __appendToFilename(config['out'], f"b_prepost_{blend}") if '|' in blendModes else f"{config['out']}_prepost"
        iio.imwrite(name, cb)
      seed = config['seed'] if 'seed' in config else int(time.time())
      ff = FilterFactory(seed)
      ch = ChannelData(cb.copy(), 'processed', 1, [], [])
      for pf in config['postFilters']:
        print(f"Post processing: {pf['name']}")
        ff.buildFilter(pf, imageData)(ch)
      cb = ch.img
      
    name = __appendToFilename(config['out'], f"b_{blend}") if '|' in blendModes else config['out']
    iio.imwrite(name, cb)
    
  return

if __name__ == '__main__':
  sys.exit(main())