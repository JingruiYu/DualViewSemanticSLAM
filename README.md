# DualViewSemanticSLAM

DualViewSemanticSLAM is a Visual Simultaneous Localization and Mapping (SLAM) system that integrates both bird's eye and front views for enhanced autonomous navigation and mapping capabilities.

## Main Functionality

- **Initial Mono Performance**: The system began with a strong performance using only the front mono view.
- **Enhanced Point Collection**: Modifications were made to collect more points from the bird's eye view by not applying certain masks.
- **Motion Model Prior**: Incorporation of a motion model prior to reduce the number of lost matches in experiments.
- **Point Integration**: Both front and bird's eye view points are utilized for a more robust mapping.
- **Pose Graph Optimization**: Implementation of pose graph optimization in local mapping and global bundle adjustment to maintain tracking stability.
- **Re-initialization Strategies**: Two re-initialization methods were introduced to recover from lost tracking, with the second method showing successful results.
