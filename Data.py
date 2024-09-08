class PixelData(object):
  def __init__(self, x, y, colour):
    self.x = x
    self.y = y
    self.colour = colour

class ChannelData(object):
  def __init__(self, img, name, weight, ops, filters):
    self.img = img
    self.weight = weight
    self.name = name
    self.filters = filters
    self.ops = ops
    
class ImageData(object):
  def __init__(self, width, height):
    self.width = width
    self.height = height