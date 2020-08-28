#include "edge_filter_ipm.h"

namespace edge_filter {

// view # 0, 1, 2, 3 -> front, rear, left, right
// mask intensity for each view
static const vector<int> VIEW_INTENSITIES = {50, 100, 150, 200};

// focal point coordinates, same order as above
static const vector<int> FOCAL_POINT_X = {189, 187, 164, 217};
static const vector<int> FOCAL_POINT_Y = {128, 248, 178, 176};

// ray from focal
static const vector<double> RAY_CENTER = {-90.0, 90.0, 180.0,
                                          0.0}; // in image frame

// vehicle param
static const double vehicle_length = 4.63;
static const double vehicle_width = 1.901;
static const double rear_axle_to_center = 1.393;
static const double pixel2meter = 0.03984;

// load sample data for testing
bool EdgeFilterIPM::load_sample_data(const string data_folder) {

  // read images
  birdseye_img_ = imread(data_folder + "birdseye.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  freespace_img_ =
      imread(data_folder + "freespace.jpg", CV_LOAD_IMAGE_GRAYSCALE);

  if (birdseye_img_.empty() || freespace_img_.empty()) {
    cerr << "Cannot load sample images from folder: " << data_folder << endl;
    return false;
  }

  // read view mask and split it into four
  view_mask_ = imread(data_folder + "view_mask.png", CV_LOAD_IMAGE_GRAYSCALE);
  if (view_mask_.empty()) {
    cerr << "Cannot load view mask from path: " << data_folder + "view_mask.png"
         << endl;
    return false;
  }

  split_view_mask();

  return true;
}

// set input data
void EdgeFilterIPM::set_input_data(const Mat &birdseye_img,
                                   const Mat &freespace_img) {
  birdseye_img_ = birdseye_img.clone();
  if (birdseye_img_.channels() != 1)
    cvtColor(birdseye_img_, birdseye_img_, COLOR_BGR2GRAY);
  freespace_img_ = freespace_img.clone();
  if (freespace_img_.channels() != 1)
    cvtColor(freespace_img_, freespace_img_, COLOR_BGR2GRAY);
}

// set view mask
bool EdgeFilterIPM::set_view_mask(const Mat &view_mask) {
  // config view masks if the mask is not empty
  if (!view_mask.empty()) {

    view_mask_ = view_mask.clone();
    if (view_mask_.channels() != 1)
      cvtColor(view_mask_, view_mask_, COLOR_BGR2GRAY);

    split_view_mask();
    return true;

  } else {
    cerr << "Warning: view mask is empty!" << endl;
    return false;
  }
}

// split the view mask image for each view
void EdgeFilterIPM::split_view_mask() {
cout << "some thing " << endl;
  Mat white_image(view_mask_.rows, view_mask_.cols, CV_8UC1);
  white_image = Scalar(255);

  // split into different views
  for (size_t i = 0; i < 4; ++i) {
    Mat tmp_mask;
    white_image.copyTo(tmp_mask, view_mask_ == Scalar(VIEW_INTENSITIES[i]));
    view_masks_.push_back(tmp_mask.clone());
  }

  // generate view index for each pixel
  view_mask_index_ = white_image.clone();
  for (int row = 0; row < view_mask_.rows; ++row)
    for (int col = 0; col < view_mask_.cols; ++col) {

      // assign view index
      for (size_t i = 0; i < 4; ++i) {
        if (view_mask_.at<uchar>(row, col) == VIEW_INTENSITIES[i]) {
          view_mask_index_.at<uchar>(row, col) = i;
          break;
        }
      }
    }
}

// process and return the filtered edge image
// method: 0 - ray-based method, 1 - line-based method
void EdgeFilterIPM::process(Mat &filtered_edge, int method) {

  // smooth the image
  Mat birdseye_img_blur;
  blur(birdseye_img_, birdseye_img_blur, Size(3, 3));

  // detect edges
  // -- auto threshold
  // Ref: https://stackoverflow.com/a/16047590/13719685
  Mat tmp_img;
  double otsu_thresh_val = threshold(birdseye_img_blur, tmp_img, 0, 255,
                                     CV_THRESH_BINARY | CV_THRESH_OTSU);
  double high_thresh_val = otsu_thresh_val,
         lower_thresh_val = otsu_thresh_val * 0.5;
  Canny(birdseye_img_blur, birdseye_edge_, lower_thresh_val, high_thresh_val);
  // cout << "canny edge thresholds: " << lower_thresh_val << ','
  //      << high_thresh_val << endl;

  Canny(freespace_img_, freespace_edge_, 50, 100);

  // apply freespace mask
  // -- erode
  Mat freespace_mask = freespace_img_ > 50;
  int erosion_size = 3;
  Mat element = getStructuringElement(
      MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1),
      Point(erosion_size, erosion_size));
  erode(freespace_mask, freespace_mask, element);

  // -- mask
  Mat birdseye_edge_free;
  birdseye_edge_.copyTo(birdseye_edge_free, freespace_mask);

  // apply view mask
  Mat birdseye_edge_masked, freespace_edge_masked;
  birdseye_edge_free.copyTo(birdseye_edge_masked, view_mask_ > 0);
  freespace_edge_.copyTo(freespace_edge_masked, view_mask_ > 0);

  // TODO: simplify freespace edge

  // remove ipm edges
  switch (method) {
  case 0:
    remove_ipm_edge_by_rays("birdseye_edge", birdseye_edge_masked,
                            birdseye_edge_filtered_, 3, 60, 2);
    remove_ipm_edge_by_rays("freespace_edge", freespace_edge_masked,
                            freespace_edge_filtered_, 5, 30, 2);
    // remove small edges
    remove_small_edges(birdseye_edge_filtered_, 20);
    remove_small_edges(freespace_edge_filtered_, 6);
    break;
  case 1:
    remove_ipm_edge_by_lines("birdseye_edge", birdseye_edge_masked,
                             birdseye_edge_filtered_);
    remove_ipm_edge_by_lines("freespace_edge", freespace_edge_masked,
                             freespace_edge_filtered_, 10.0);
    // remove small edges
    remove_small_edges(birdseye_edge_filtered_, 20);
    remove_small_edges(freespace_edge_filtered_, 6);

    break;
  case 2:
    remove_ipm_edge_by_lines("birdseye_edge", birdseye_edge_masked,
                             birdseye_edge_filtered_, 15.0);
    remove_ipm_edge_by_rays("freespace_edge", freespace_edge_masked,
                            freespace_edge_filtered_, 5, 30, 2);
    // remove small edges
    remove_small_edges(birdseye_edge_filtered_, 20);
    remove_small_edges(freespace_edge_filtered_, 6);

    break;
  case 3:
    remove_ipm_edge_by_contour_orientation("birdseye_edge",
                                           birdseye_edge_masked,
                                           birdseye_edge_filtered_, 3, 15, 20);
    remove_ipm_edge_by_contour_orientation("freespace_edge",
                                           freespace_edge_masked,
                                           freespace_edge_filtered_, 5, 15, 20);
    // remove small edges
    remove_small_edges(birdseye_edge_filtered_, 20);
    break;
  default:
    cout << "Invalid method index for filtering ipm edges!" << endl;
    return;
  }

  // combine edges
  addWeighted(birdseye_edge_filtered_, 0.5, freespace_edge_filtered_, 1.0, 0.0,
              merged_edge_);
  filtered_edge = merged_edge_.clone();
}

// remove ipm edges with rays passing the focal points
void EdgeFilterIPM::remove_ipm_edge_by_rays(string edge_name, Mat &edge,
                                            Mat &filtered_edge,
                                            double angle_step,
                                            int max_ipm_edge_pixel_num,
                                            int min_ipm_edge_pixel_num) {
  // set param
  int angle_num = ceil(360.0 / angle_step) + 1;

  // create buffer
  vector<vector<vector<Point>>> view_bin_edge_pixel(4);
  for (size_t i = 0; i < view_bin_edge_pixel.size(); ++i)
    view_bin_edge_pixel[i].resize(angle_num);

  filtered_edge = Mat::zeros(edge.rows, edge.cols, CV_8UC1);

  // iterate the edge
  for (int row = 0; row < edge.rows; ++row)
    for (int col = 0; col < edge.cols; ++col) {

      if (edge.at<uchar>(row, col) == 0)
        continue;

      // decide view index
      int view_ind = view_mask_index_.at<uchar>(row, col);
      if (view_ind >= VIEW_INTENSITIES.size())
        continue;

      // compute angle and index
      double angle =
          atan2(row - FOCAL_POINT_Y[view_ind], col - FOCAL_POINT_X[view_ind]);
      angle = fmod(angle + 2 * M_PI, 2 * M_PI) * 180.0 / M_PI; // degree
      int angle_index = ceil(angle / angle_step);

      // add to buffer
      view_bin_edge_pixel[view_ind][angle_index].push_back(Point(col, row));
    }

  // thresholding to only preserve small bins
  for (auto view_bin : view_bin_edge_pixel)
    for (auto bin : view_bin) {
      if (min_ipm_edge_pixel_num < bin.size() &&
          bin.size() < max_ipm_edge_pixel_num) {
        for (auto p : bin)
          filtered_edge.at<uchar>(p) = 255;
      }
    }

  // show
  imshow(edge_name, filtered_edge);
  waitKey(1);
}

// remove ipm edges with detected lines passing the focal points
void EdgeFilterIPM::remove_ipm_edge_by_lines(string edge_name, Mat &edge,
                                             Mat &filtered_edge,
                                             float focal_point_range_thresh) {

  // find focal points by line detection, using: Probabilistic Line Transform
  vector<Vec4i> linesP; // will hold the results of the detection
  HoughLinesP(edge, linesP, 1, CV_PI / 180, 20, 15,
              10); // runs the actual detection

  // show detected lines
  Mat line_img;
  cvtColor(edge, line_img, COLOR_GRAY2BGR);
  for (size_t i = 0; i < linesP.size(); ++i) {
    Vec4i l = linesP[i];
    line(line_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2,
         LINE_8);
  }
  imshow(edge_name, line_img);
  waitKey(1);

  // filter edges by detected lines
  filtered_edge = edge.clone();

  for (size_t i = 0; i < linesP.size(); ++i) {

    // decide view index
    int view_ind_for_cur_line = -1;
    Vec4i l = linesP[i];

    int view_ind_1 = view_mask_index_.at<uchar>(Point(l[0], l[1]));
    int view_ind_2 = view_mask_index_.at<uchar>(Point(l[2], l[3]));

    if (view_ind_1 < VIEW_INTENSITIES.size() &&
        view_ind_2 < VIEW_INTENSITIES.size() && view_ind_1 == view_ind_2)
      view_ind_for_cur_line = view_ind_1;
    else {
      // cout << "Warning: Current line is not in any view!" << endl;
      continue;
    }

    // check distance to focal point
    Point focal_point = Point(FOCAL_POINT_X[view_ind_for_cur_line],
                              FOCAL_POINT_Y[view_ind_for_cur_line]);
    double dist = point_to_line_distance(focal_point, l);
    if (dist < focal_point_range_thresh) {
      // clear the edge
      line(filtered_edge, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0), 3,
           LINE_8);
    }
  }
}

// remove ipm edges with thresholding the contour orientation to the focal point
void EdgeFilterIPM::remove_ipm_edge_by_contour_orientation(
    string edge_name, Mat &edge, Mat &filtered_edge, int approx_epsilon,
    double angle_threshold, double min_ipm_edge_pixel_num,
    bool use_approx_edge) {

  // get contours
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(edge, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE,
               Point(0, 0));

  // check if use the approximated edges
  if (use_approx_edge) {
    filtered_edge = Mat::zeros(edge.rows, edge.cols, CV_8UC1);

    // draw approximated edges
    for (size_t i = 0; i < contours.size(); ++i) {
      vector<Point> contour_fine;
      approxPolyDP(contours[i], contour_fine, 2, true);
      vector<vector<Point>> contours_fine;
      contours_fine.push_back(contour_fine);
      drawContours(filtered_edge, contours_fine, 0, Scalar(255), 1, 8,
                   hierarchy, 0, Point());
    }
  } else
    filtered_edge = edge.clone();

  // images for visualization
  Mat results_vis;
  cvtColor(edge, results_vis, COLOR_GRAY2BGR);
  Mat edge_clean_vis = results_vis.clone();

  // iterate the extracted contours
  for (size_t i = 0; i < contours.size(); ++i) {

    // check if the contour is too short
    if (contours[i].size() > min_ipm_edge_pixel_num) {

      // simplify the contour
      vector<Point> contour_coarse;
      approxPolyDP(contours[i], contour_coarse, approx_epsilon, true);

      // check if the oriention angle exceeds the threshold
      for (size_t j = 0; j < contour_coarse.size() - 1; ++j) {
        int view_ind = view_mask_index_.at<uchar>(contour_coarse[j]);

        Point mid_point((contour_coarse[j].x + contour_coarse[j + 1].x) * 0.5,
                        (contour_coarse[j].y + contour_coarse[j + 1].y) * 0.5);

        Point ray_mid_point(FOCAL_POINT_X[view_ind] - mid_point.x,
                            FOCAL_POINT_Y[view_ind] - mid_point.y);

        Point line_seg_half(contour_coarse[j].x - mid_point.x,
                            contour_coarse[j].y - mid_point.y);

        double angle_distance = angle_dist(ray_mid_point, line_seg_half);
        angle_distance = min(angle_distance, M_PI - angle_distance);

        if (2 * norm(line_seg_half) > 0.5 * min_ipm_edge_pixel_num &&
            angle_distance < angle_threshold / 180.0 * M_PI) {

          // clear the line
          line(filtered_edge, contour_coarse[j], contour_coarse[j + 1],
               Scalar(0), 2 * approx_epsilon, LINE_8);

          //  draw for visualization
          line(edge_clean_vis, contour_coarse[j], contour_coarse[j + 1],
               Scalar(0, 0, 255), 2 * approx_epsilon, LINE_8);
        } else {
          //  draw for visualization
          line(edge_clean_vis, contour_coarse[j], contour_coarse[j + 1],
               Scalar(0, 255, 0), 2 * approx_epsilon, LINE_8);
        }

        // draw ray to middle point of the line segment
        line(edge_clean_vis, mid_point,
             Point(FOCAL_POINT_X[view_ind], FOCAL_POINT_Y[view_ind]),
             Scalar(255, 0, 0), 1, LINE_8);

        // draw terminal points of the line segment
        circle(edge_clean_vis, contour_coarse[j], approx_epsilon,
               Scalar(0, 255, 255), -1);
        circle(edge_clean_vis, contour_coarse[j + 1], approx_epsilon,
               Scalar(0, 255, 255), -1);
      }
    } else {
      // clear small edge
      drawContours(filtered_edge, contours, i, Scalar(0), 1, 8, hierarchy, 0,
                   Point());
      drawContours(edge_clean_vis, contours, i, Scalar(0, 0, 255), 1, 8,
                   hierarchy, 0, Point());
    }
  }

  // draw focal points
  draw_focal_points(edge_clean_vis);

  // show
  addWeighted(results_vis, 0.5, edge_clean_vis, 0.5, 0.0, results_vis);
  imshow(edge_name, results_vis);
  waitKey(1);
}

// remove small edges by a threshold
void EdgeFilterIPM::remove_small_edges(Mat &edge, int edge_length_thresh) {

  Mat edge_without_small_edge = Mat::zeros(edge.size(), CV_8UC1);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(edge, contours, hierarchy, CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE, Point(0, 0));

  for (int i = 0; i < contours.size(); i++) {
    if (contours[i].size() > edge_length_thresh)
      drawContours(edge_without_small_edge, contours, i, Scalar(255), 1, 8,
                   hierarchy, 0, Point());
  }

  // update original edge
  edge = edge_without_small_edge;
}

// draw focal points
void EdgeFilterIPM::draw_focal_points(Mat &img) {

  for (int i = 0; i < 4; ++i) {
    circle(img, Point(FOCAL_POINT_X[i], FOCAL_POINT_Y[i]), 2, Scalar(255, 0, 0),
           -1);
    circle(img, Point(FOCAL_POINT_X[i], FOCAL_POINT_Y[i]), 6, Scalar(0, 0, 255),
           1);
  }
}

// show focal points
void EdgeFilterIPM::show_focal_points(int interval_time) {

  Mat focal_points_img;
  cvtColor(birdseye_edge_, focal_points_img, CV_GRAY2BGR);
  draw_focal_points(focal_points_img);
  imshow("focal_points", focal_points_img);
  waitKey(interval_time);
}

// draw rays from focal points
void EdgeFilterIPM::draw_rays_from_focal(Mat &img) {

  Mat ray_img = img.clone();
  double angle_step = 3.0;  // degree
  double angle_start = 0.0; // degree
  double angle_end = 360.0; // degree
  double radius = 300.0;    // pixel

  for (int i = 0; i < 4; ++i) {

    Mat tmp_ray_img = ray_img.clone();

    // generate rays
    double cur_angle = angle_start;

    while (cur_angle < angle_end) {

      Point start(FOCAL_POINT_X[i], FOCAL_POINT_Y[i]);
      Point end = start + Point(int(radius * cos(cur_angle / 180.0 * M_PI)),
                                int(radius * sin(cur_angle / 180.0 * M_PI)));

      line(tmp_ray_img, start, end, Scalar(255, 0, 0), 1, LINE_AA);

      cur_angle += angle_step;
    }

    // apply view mask
    tmp_ray_img.copyTo(ray_img, view_masks_[i]);
  }

  addWeighted(img, 0.7, ray_img, 0.3, 0.0, img);
}

// show rays from focal points
void EdgeFilterIPM::show_rays_from_focal(int interval_time) {

  Mat ray_image;
  cvtColor(birdseye_img_, ray_image, CV_GRAY2BGR);

  // draw rays from focal points
  draw_rays_from_focal(ray_image);

  // draw focal points
  draw_focal_points(ray_image);

  imshow("ray_image", ray_image);
  waitKey(interval_time);
}

// show raw edge on birdseye image
void EdgeFilterIPM::show_raw_edge_on_birdseye_image(int interval_time) {

  Mat raw_edge_on_birdseye_image;
  cvtColor(birdseye_img_, raw_edge_on_birdseye_image, CV_GRAY2BGR);

  // draw edges
  for (int row = 0; row < birdseye_img_.rows; ++row)
    for (int col = 0; col < birdseye_img_.cols; ++col)
      if (birdseye_edge_.at<uchar>(row, col) > 0)
        raw_edge_on_birdseye_image.at<Vec3b>(row, col) = Vec3b(0, 255, 0);
      else if (freespace_edge_.at<uchar>(row, col) > 0)
        raw_edge_on_birdseye_image.at<Vec3b>(row, col) = Vec3b(0, 0, 255);

  // draw focal points
  draw_focal_points(raw_edge_on_birdseye_image);

  imshow("raw_edge_on_birdseye_image", raw_edge_on_birdseye_image);
  waitKey(interval_time);
}

// show filtered edge on birdseye image
void EdgeFilterIPM::show_filtered_edge_on_birdseye_image(int interval_time) {

  Mat filtered_edge_on_birdseye_image;
  cvtColor(birdseye_img_, filtered_edge_on_birdseye_image, CV_GRAY2BGR);

  // draw edges
  for (int row = 0; row < birdseye_img_.rows; ++row)
    for (int col = 0; col < birdseye_img_.cols; ++col)
      if (birdseye_edge_filtered_.at<uchar>(row, col) > 0)
        filtered_edge_on_birdseye_image.at<Vec3b>(row, col) = Vec3b(0, 255, 0);
      else if (freespace_edge_filtered_.at<uchar>(row, col) > 0)
        filtered_edge_on_birdseye_image.at<Vec3b>(row, col) = Vec3b(0, 0, 255);

  // draw focal points
  draw_focal_points(filtered_edge_on_birdseye_image);

  // draw rays
  draw_rays_from_focal(filtered_edge_on_birdseye_image);

  imshow("filtered_edge_on_birdseye_image", filtered_edge_on_birdseye_image);
  waitKey(interval_time);
}

// save edge to file
void EdgeFilterIPM::save_edge_to_file(string file_path) {
  imwrite(file_path, merged_edge_);
}

// convert edge to pcl labeled cloud
void EdgeFilterIPM::convert_edge_to_pcl_cloud(SemanticCloud::Ptr cloud) {

  // convert pixels to points
  for (int row = 0; row < merged_edge_.rows; ++row)
    for (int col = 0; col < merged_edge_.cols; ++col) {

      // set label
      int label = -1;
      if (merged_edge_.at<uchar>(row, col) == 0)
        continue; // free
      else if (merged_edge_.at<uchar>(row, col) < 129)
        label = 0; // edge
      else
        label = 1; // freespace

      SemanticPoint point;
      point.x =
          (merged_edge_.rows / 2 - row) * pixel2meter + rear_axle_to_center;
      point.y = (merged_edge_.cols / 2 - col) * pixel2meter;
      point.z = 0.0;
      point.label = label;

      cloud->points.push_back(point);
    }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
}
}