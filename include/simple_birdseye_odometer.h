#ifndef SIMPLE_BIRDSEYE_ODOMETER_H
#define SIMPLE_BIRDSEYE_ODOMETER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pclomp/ndt_omp.h>

#include "edge_filter_ipm.h"

namespace birdseye_odometry {

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace edge_filter;

typedef pcl::PointXYZL SemanticPoint;
typedef pcl::PointCloud<pcl::PointXYZL> SemanticCloud;

// vehicle param
static const double vehicle_length = 4.63;
static const double vehicle_width = 1.901;
static const double rear_axle_to_center = 1.393;
static const double pixel2meter = 0.03984;

class SimpleBirdseyeOdometer {

public:
  SimpleBirdseyeOdometer();
  void config_edge_filter(const Mat &view_mask);
  void add_new_frame(int index, const Mat &birdseye_img,
                     const Mat &freespace_img);
  Matrix4f get_current_pose();
  SemanticCloud::Ptr get_local_cloud();
  SemanticCloud::Ptr get_global_cloud(float voxel_size);
  vector<Matrix4f> get_trajectory();

private:
  void config_viewer();
  double estimate_relative_motion(SemanticCloud::Ptr current_cloud,
                                  SemanticCloud::Ptr aligned_cloud,
                                  Matrix4f &relative_pose);
  void update_local_cloud(SemanticCloud::Ptr aligned_cloud,
                          bool update_key_cloud,
                          bool with_local_update = false);
  void update_global_cloud();

  // poses
  Matrix4f current_pose_, local_cloud_pose_;
  vector<Matrix4f> trajectory_;
  double good_alignment_thresh_;

  // clouds
  vector<bool> enabled_label_;
  SemanticCloud::Ptr current_cloud_;
  SemanticCloud::Ptr local_cloud_, global_cloud_;
  int key_cloud_index_;
  double key_cloud_range_threshold_, key_cloud_angle_threshold_;
  vector<SemanticCloud::Ptr> clouds_;

  // edge filter
  shared_ptr<EdgeFilterIPM> edge_filter_ptr_;
  Mat current_edge_filtered_;

  // aligner
  pclomp::NormalDistributionsTransform<SemanticPoint, SemanticPoint>::Ptr
      ndt_aligner_ptr_;

  // viewer
  pcl::visualization::PCLVisualizer::Ptr viewer_ptr_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}

#endif // !SIMPLE_BIRDSEYE_ODOMETER_H