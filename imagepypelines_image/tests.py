import imagepypelines as ip
import numpy as np

# make sure the image plugin is installed
ip.require("image")




tasks = {
        'geckos':ip.Input(0),
        # normalize the inputs
        'float_geckos':(ip.image.CastTo(np.float64), 'geckos'),
        'normalized_geckos': (ip.image.NormAB(0,255), 'float_geckos'),
        'display_safe' : (ip.image.DisplaySafe(), 'normalized_geckos'),
        # split into RGB channels
        ('red','green','blue') : (ip.image.ChannelSplit(), 'display_safe'),
        }

pipeline = ip.Pipeline(tasks, name='Lenna')

geckos = [ip.image.gecko() for i in range(10)]
# processed = pipeline.process(geckos)


bad_processed = pipeline.process([np.random.rand(512,512)] )
bad_processed = pipeline.process([np.random.rand(512,512)], skip_checks=True )


import pdb; pdb.set_trace()
