import torch
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt

   
def plot_samples(samples, 
                 met        = {}, 
                 is_gt      = True, 
                 predictor  = None,
                 name_label = 'gt'):
    n = len(samples)
    nrows = int(-(-n/3)) # ceil
    ncols = 3
    fig, axs = plt.subplots(nrows   = nrows, 
                          ncols   = ncols, 
                          figsize = (21, 7))
    for i,s in enumerate(samples):
        row = i//ncols
        col = i%ncols
        ax = axs[row][col] if len(axs.shape)==2 else axs[i]
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:,:, ::-1], metadata=met, scale=0.5)
        if is_gt:
            # visualize ground-truths
            v = v.draw_dataset_dict(s)
        else:
            # predict
            outputs = predictor(img)
            # visualize prediction results
            instances = outputs["instances"].to("cpu")
            v = v.draw_instance_predictions(instances)

        ax.imshow(v.get_image())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(name_label + '.png')
    plt.show()


def plot_random_samples(name_ds, n=3, predictor=None):
    # access
    ds = DatasetCatalog.get(name_ds)
    met = MetadataCatalog.get(name_ds)
    samples = random.sample(ds, n)
    # plot samples with ground-truths
    plot_samples(samples, met)
    # plot predictions
    plot_samples(samples, 
                 met        = met, 
                 predictor  = predictor, 
                 is_gt      = False,
                 name_label = 'pred')