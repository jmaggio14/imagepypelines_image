import imagepypelines as ip
import numpy as np

# make sure the image plugin is installed
ip.require("image")


ip.MASTER_LOGGER.setLevel(10)
ip.MASTER_LOGGER.debug("defining our tasks")

tasks = {
        'geckos':ip.Input(0),
        # normalize the inputs
        'float_geckos':(ip.image.CastTo(np.float64), 'geckos'),
        'normalized_geckos': (ip.image.NormAB(0,255), 'float_geckos'),
        'display_safe' : (ip.image.DisplaySafe(), 'normalized_geckos'),
        # split into RGB channels
        ('red','green','blue') : (ip.image.ChannelSplit(), 'display_safe'),
        }

ip.MASTER_LOGGER.debug("make our pipeline")
pipeline = ip.Pipeline(tasks, name='Lenna')

geckos = [ip.image.gecko() for i in range(10)]
# processed = pipeline.process(geckos)

ip.MASTER_LOGGER.debug("processing our data")
bad_processed = pipeline.process([np.random.rand(512,512)] )

ip.MASTER_LOGGER.debug("this will now throw an error")
bad_processed = pipeline.process([np.random.rand(512,512)], skip_checks=True )
