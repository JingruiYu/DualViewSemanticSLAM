#ifndef EDGE_FILTER_IPM_H
#define EDGE_FILTER_IPM_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

typedef pcl::PointXYZL SemanticPoint;
typedef pcl::PointCloud<pcl::PointXYZL> SemanticCloud;

namespace edge_filter {

class EdgeFilterIPM {

public:
  bool load_sample_data(const string data_folder);
  void set_input_data(const Mat &birdseye_img, const Mat &freespace_img);
  bool set_view_mask(const Mat &view_mask = Mat());
  void process(Mat &filtered_edge, int method = 0);
  void draw_focal_points(Mat &img);
  void show_focal_points(int interval_time);
  void draw_rays_from_focal(Mat &img);
  void show_rays_from_focal(int interval_time);
  void show_raw_edge_on_birdseye_image(int interval_time);
  void show_filtered_edge_on_birdseye_image(int interval_time);
  void save_edge_to_file(string file_path);
  void convert_edge_to_pcl_cloud(SemanticCloud::Ptr cloud);

private:
  void split_view_mask();
  void remove_ipm_edge_by_lines(string edge_name, Mat &edge, Mat &filtered_edge,
                                float focal_point_range_thresh = 5.0);
  void remove_ipm_edge_by_rays(string edge_name, Mat &edge, Mat &filtered_edge,
                               double angle_step, int max_ipm_edge_pixel_num,
                               int min_ipm_edge_pixel_num);
  void remove_ipm_edge_by_contour_orientation(string edge_name, Mat &edge,
                                              Mat &filtered_edge,
                                              int approx_epsilon,
                                              double angle_threshold,
                                              double min_ipm_edge_pixel_num,
                                              bool use_approx_edge = true);
  void remove_small_edges(Mat &edge, int contour_length_thresh);

  Mat birdseye_img_;
  Mat freespace_img_;
  Mat view_mask_;
  Mat view_mask_index_;
  vector<Mat> view_masks_;

  Mat birdseye_edge_, freespace_edge_;
  Mat birdseye_edge_filtered_, freespace_edge_filtered_;
  Mat merged_edge_;
};

// functions used to compute point-to-line distance
static std::array<int, 3> cross(const std::array<int, 3> &a,
                                const std::array<int, 3> &b) {
  std::array<int, 3> result;
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
  return result;
}

static double point_to_line_distance(const cv::Point &p,
                                     const cv::Vec4i &line) {
  std::array<int, 3> pa{{line[0], line[1], 1}};
  std::array<int, 3> pb{{line[2], line[3], 1}};
  std::array<int, 3> l = cross(pa, pb);
  return std::abs((p.x * l[0] + p.y * l[1] + l[2])) * 1.0 /
         std::sqrt(double(l[0] * l[0] + l[1] * l[1]));
}

// functions used to compute the angle distance of two vectors
static double angle_dist(const Point &v1, const Point &v2) {
  double cosAngle = v1.dot(v2) / (cv::norm(v1) * cv::norm(v2));
  if (cosAngle > 1.0)
    return 0.0;
  else if (cosAngle < -1.0)
    return M_PI;
  return std::acos(cosAngle);
}
}

#endif // !EDGE_FILTER_IPM_H