# RDPC
## Instructions for use
You just need to edit main.py and run it.
<pre><code>data = np.loadtxt('aggregation.txt') # aggregation.txt can be relpacecd by any other dataset under the 'datasets' folder. 
RDPC_clustering = RDPC.RDPC(metric='mutual')  # you can select the mutual reachability distance as the metric (metric='mutual'), or select Euclidean distance as the metric (metric='euclidean').
result = RDPC_clustering.fit_predict(data)  # RDPC will generate a decision graph. In the decision graph, you select objects with $\delta_{rdpc}$ values significantly larger than the surrounding objects as maximum objects by rectangle box, and then RDPC will return the clustering result
</code></pre>
