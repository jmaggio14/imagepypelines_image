import imagepypelines as ip
import numpy as np

# make sure the image plugin is installed
ip.require_plugin("image")




tasks = {
        'lennas':ip.Input(0),
        'float_lennas':(ip.image.CastTo(np.float64), 'lennas'),
        'normalized_lennas': (ip.image.NormAB(0,255), 'float_lennas'),
        'display_safe' : (ip.image.DisplaySafe(), 'normalized_lennas'),
        }

pipeline = ip.Pipeline(tasks, name='Lenna')

lennas = [ip.lenna() for i in range(10)]
processed = pipeline.process(lennas)


import pdb; pdb.set_trace()
