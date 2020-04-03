import imagepypelines as ip
import numpy as np

# make sure the image plugin is installed
ip.require("image")




tasks = {
        'lennas':ip.Input(0),
        # normalize the inputs
        'float_lennas':(ip.image.CastTo(np.float64), 'lennas'),
        'normalized_lennas': (ip.image.NormAB(0,255), 'float_lennas'),
        'display_safe' : (ip.image.DisplaySafe(), 'normalized_lennas'),
        # split into RGB channels
        ('red','green','blue') : (ip.image.ChannelSplit(), 'display_safe'),
        }

pipeline = ip.Pipeline(tasks, name='Lenna')

lennas = [ip.lenna() for i in range(10)]
# processed = pipeline.process(lennas)


bad_processed = pipeline.process([np.random.rand(512,512)] )
bad_processed = pipeline.process([np.random.rand(512,512)], skip_checks=True )


import pdb; pdb.set_trace()
