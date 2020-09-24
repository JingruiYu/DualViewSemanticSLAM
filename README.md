Using the bird view and front view for SLAM

v0.0 is the original, Only Front Mono, which have good performance.

v0.1:
in mono_Bird_sem.cc, the mask of birdview not applied, for get much more points. 122~124

v0.2:
The Twb_c of both GroundTruth and pose after the TrackWithLocalMap are drawn.

v0.3:
The norm is 0.4, the inited Frame is 44. But the performance is not stable. The norm is 0.6 is worst.

v0.4:
The prior of motion model is added. The number of matches and lost is less in 7 experiment.

v0.5:
Both the front and bird points are added.

v0.6:
The bird points are removed. The pose graph are added in both localMapping and GlobalBA. The weight of PoseGraph should not larger than 2e3.
If the weight of PoseGraph is larger, tracking will be lost. So, the rebuilt map is needed for tracking lost.
Check why the pure front slam don't have good shape?

v0.7:
Two kinds of re-initialization were implemented.
First is re-init by the prior pose.
Second is re-init by both Fundamental and the prior pose.
Now using the second method, the lost can be find successfully.
While the loop closing have problem, still need to debug. Cannot find.