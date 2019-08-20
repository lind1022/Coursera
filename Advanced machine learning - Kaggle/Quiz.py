import numpy as np
from sklearn.metrics import log_loss
import scipy.stats as stats

pred = np.full((100000), 0.3)



means = np.arange(0.771, 0.773, step=0.0001).tolist()

for mean in means:
    actual_pt1 = np.full((round(100000*mean)), 1)
    actual_pt2 = np.full((round(100000*(1-mean))), 0)
    actual = np.concatenate((actual_pt1, actual_pt2))
    score = log_loss(actual, pred)
    print(str(mean) + ":" + str(score))



log_loss(actual, pred)
