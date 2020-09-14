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