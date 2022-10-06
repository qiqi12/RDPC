import numpy as np
import RDPC


data = np.loadtxt('aggregation.txt')
RDPC_clustering = RDPC.RDPC(metric='mutual')
result = RDPC_clustering.fit_predict(data)




