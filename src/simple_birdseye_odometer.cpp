#include "simple_birdseye_odometer.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace birdseye_odometry {

// constructor
SimpleBirdseyeOdometer::SimpleBirdseyeOdometer()
    : local_cloud_(new SemanticCloud), global_cloud_(new SemanticCloud),
      current_cloud_(new SemanticCloud),
      ndt_aligner_ptr_(
          new pclomp::NormalDistributionsTransform<SemanticPoint,
                                                   SemanticPoint>()),
      viewer_ptr_(new pcl::visualization::PCLVisualizer("viewer")) {
  // init pose
  current_pose_.setIdentity();

  // settings
  // -- enabled label
  enabled_label_ = {true, false};

  // -- threshold to accept new cloud
  good_alignment_thresh_ = 2.0;

  // -- threshold for updating the key cloud
  key_cloud_range_threshold_ = 1;                // m
  key_cloud_angle_threshold_ = 5.0 / 180.0 * M_PI; // rad

  // create edge filter
  edge_filter_ptr_ = make_shared<EdgeFilterIPM>();
  cout << "do we use here?" << endl;
  // set NDT aligner
  ndt_aligner_ptr_->setStepSize(0.1);
  ndt_aligner_ptr_->setResolution(1.0);
  ndt_aligner_ptr_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  ndt_aligner_ptr_->setNumThreads(2);

  // set viewer
  config_viewer();
}

// config edge filter
void SimpleBirdseyeOdometer::config_edge_filter(const Mat &view_mask) {
  edge_filter_ptr_->set_view_mask(view_mask);
}

// config viewer
void SimpleBirdseyeOdometer::config_viewer() {
  {
    // config viewport for global, local and current cloud
    int v1(0);
    viewer_ptr_->createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    viewer_ptr_->createViewPortCamera(v1);
    viewer_ptr_->setBackgroundColor(0, 0, 0, v1);
    viewer_ptr_->addPointCloud<SemanticPoint>(local_cloud_, "local_cloud", v1);

    SemanticCloud::Ptr current_cloud(new SemanticCloud);
    viewer_ptr_->addPointCloud<SemanticPoint>(current_cloud, "current_cloud",
                                              v1);

    viewer_ptr_->addPointCloud<SemanticPoint>(global_cloud_, "global_cloud",
                                              v1);

    viewer_ptr_->addCube(-(vehicle_length / 2 - rear_axle_to_center),
                         vehicle_length / 2 + rear_axle_to_center,
                         -vehicle_width / 2, vehicle_width / 2, 0.0, 0.2, 128.0,
                         69.0, 0.0, "vehicle", v1);
    viewer_ptr_->addCoordinateSystem(1.0, 0.0, 0.0, 0.3, "vehicle_frame", v1);
    viewer_ptr_->addCoordinateSystem(1.0, 0.0, 0.0, 0.0, "map_frame", v1);
  }
}

// add new frame and process
void SimpleBirdseyeOdometer::add_new_frame(int index, const Mat &birdseye_img,
                                           const Mat &freespace_img) {
  // filter by IPM edge filter
  edge_filter_ptr_->set_input_data(birdseye_img, freespace_img);
  edge_filter_ptr_->process(current_edge_filtered_, 3);
  edge_filter_ptr_->show_filtered_edge_on_birdseye_image(1);

  SemanticCloud::Ptr current_cloud_raw(new SemanticCloud);
  edge_filter_ptr_->convert_edge_to_pcl_cloud(current_cloud_raw);

  // filter the label
  SemanticCloud::Ptr current_cloud_filtered(new SemanticCloud);

  for (auto p : current_cloud_raw->points) {
    if (enabled_label_[p.label])
      current_cloud_filtered->push_back(p);
  }
  current_cloud_filtered->is_dense = true;

  // update current cloud
  current_cloud_ = current_cloud_filtered;

  // check if initialized
  if (clouds_.empty()) {

    // add inital pose
    trajectory_.push_back(Matrix4f::Identity());

    // add to the cloud lists
    clouds_.push_back(current_cloud_);

    // init the key cloud index
    key_cloud_index_ = 0;

    // init the local cloud
    local_cloud_ = current_cloud_;
    local_cloud_pose_ = Matrix4f::Identity();

    // init the global cloud
    update_global_cloud();

  } else {

    // get current pose
    SemanticCloud::Ptr current_aligned_cloud(new SemanticCloud);
    Matrix4f relative_pose;
    double score = estimate_relative_motion(
        current_cloud_, current_aligned_cloud, relative_pose);

    if (score < good_alignment_thresh_) {

      // update trajectories
      current_pose_ = local_cloud_pose_ * relative_pose;
      trajectory_.push_back(current_pose_);

      cout << "current_pose: \n" << current_pose_ << endl;

      // show
      {
        { // -- global
          pcl::visualization::PointCloudColorHandlerCustom<SemanticPoint>
              global_handler(global_cloud_, 100, 100, 100);
          viewer_ptr_->updatePointCloud(global_cloud_, global_handler,
                                        "global_cloud");
        }

        { // -- local
          //---- convert to map frame
          SemanticCloud::Ptr local_cloud_in_map(new SemanticCloud);
          transformPointCloud(*local_cloud_, *local_cloud_in_map,
                              local_cloud_pose_);

          pcl::visualization::PointCloudColorHandlerCustom<SemanticPoint>
              local_handler(local_cloud_in_map, 255, 255, 255);
          viewer_ptr_->updatePointCloud(local_cloud_in_map, local_handler,
                                        "local_cloud");
          viewer_ptr_->setPointCloudRenderingProperties(
              pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "local_cloud");
        }

        { // -- current
          //---- convert to map frame
          SemanticCloud::Ptr current_cloud_in_map(new SemanticCloud);
          Matrix4f transform = trajectory_.back();
          transformPointCloud(*current_cloud_, *current_cloud_in_map,
                              transform);

          viewer_ptr_->updatePointCloud(current_cloud_in_map, "current_cloud");
          viewer_ptr_->setPointCloudRenderingProperties(
              pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
              "current_cloud");
        }

        // -- vehicle model
        Eigen::Affine3f vehicle_pose;
        vehicle_pose.matrix() = trajectory_.back();
        viewer_ptr_->updateShapePose("vehicle", vehicle_pose);
        viewer_ptr_->updateCoordinateSystemPose("vehicle_frame", vehicle_pose);

        // -- trajectory
        SemanticPoint waypoint;
        waypoint.x = vehicle_pose.translation()[0];
        waypoint.y = vehicle_pose.translation()[1];
        waypoint.z = vehicle_pose.translation()[2];
        string waypoint_name = "waypoint" + to_string(trajectory_.size());
        viewer_ptr_->addSphere(waypoint, 0.1, 150, 0, 0, waypoint_name);

        viewer_ptr_->spinOnce();
      }

      // add to the cloud list
      clouds_.push_back(current_cloud_);

      // update the local cloud
      update_local_cloud(current_aligned_cloud, false);

      // update the key cloud and global cloud if needed
      Eigen::Vector3f relative_trans = relative_pose.topRightCorner(3, 1);
      Eigen::AngleAxisf relative_rot;
      relative_rot = Eigen::Matrix3f(relative_pose.topLeftCorner(3, 3));

      if (relative_trans.norm() > key_cloud_range_threshold_ ||
          fabs(relative_rot.angle()) > key_cloud_angle_threshold_) {

        // -- update the key cloud
        key_cloud_index_ = clouds_.size() - 1;

        cout << "Key cloud updated! New key cloud index: " << key_cloud_index_
             << endl;

        // -- update the global cloud by adding the local cloud
        update_global_cloud();

        // -- update the local cloud
        update_local_cloud(current_aligned_cloud, true);

        // -- show the key waypoint
        SemanticPoint key_waypoint;
        key_waypoint.x = trajectory_.back()(0, 3);
        key_waypoint.y = trajectory_.back()(1, 3);
        key_waypoint.z = trajectory_.back()(2, 3);
        string key_waypoint_name =
            "key_waypoint" + to_string(trajectory_.size());
        viewer_ptr_->addSphere(key_waypoint, 0.2, 0, 0, 150, key_waypoint_name);

        viewer_ptr_->spinOnce();
      }

    } else {
      cout << "Warning: bad relative motion estimation with score: " << score
           << endl;
    }
  }
}

// estimate the relative motion from current cloud to local cloud
double SimpleBirdseyeOdometer::estimate_relative_motion(
    SemanticCloud::Ptr current_cloud, SemanticCloud::Ptr aligned_cloud,
    Matrix4f &relative_pose) {

  // align with NDT
  // -- set input
  Matrix4f initTransform = local_cloud_pose_.inverse() * current_pose_;
  ndt_aligner_ptr_->setInputSource(current_cloud);
  ndt_aligner_ptr_->setInputTarget(local_cloud_);

  // -- align and get the results
  ndt_aligner_ptr_->align(*aligned_cloud, initTransform);
  relative_pose = ndt_aligner_ptr_->getFinalTransformation();
  cout << "relative_pose: \n" << relative_pose << endl;
  double score = ndt_aligner_ptr_->getFitnessScore();
  cout << "score: " << score << endl;

  return score;
}

// update local cloud
void SimpleBirdseyeOdometer::update_local_cloud(
    SemanticCloud::Ptr aligned_cloud, bool update_key_cloud,
    bool with_local_update) {

  // TODO: add good points of interval clouds!

  // move local cloud to new pose if needed
  if (update_key_cloud) {

    // set new local cloud
    if (with_local_update && false) {

      // update local cloud with new key cloud
      *local_cloud_ += *aligned_cloud;

      // tranform local cloud to current key pose
      Matrix4f transform = trajectory_.back().inverse() * local_cloud_pose_;
      SemanticCloud::Ptr new_local_cloud(new SemanticCloud);
      transformPointCloud(*local_cloud_, *new_local_cloud, transform);

      // remove those points out of the view
      float view_range = 7.7; // m

      pcl::PassThrough<SemanticPoint> pass;
      pass.setInputCloud(new_local_cloud);

      SemanticCloud::Ptr filtered_local_cloud(new SemanticCloud);

      pass.setFilterFieldName("x");
      pass.setFilterLimits(-view_range, view_range);
      pass.filter(*filtered_local_cloud);

      pass.setInputCloud(filtered_local_cloud);
      pass.setFilterFieldName("y");
      pass.setFilterLimits(-view_range, view_range);
      pass.filter(*filtered_local_cloud);

      local_cloud_ = filtered_local_cloud;
    } else {

      // direct use the new key cloud
      local_cloud_ = current_cloud_;
    }

    // update the pose of local cloud
    local_cloud_pose_ = trajectory_.back();
  }
}

// update global cloud
void SimpleBirdseyeOdometer::update_global_cloud() {
  // transform to the first frame
  SemanticCloud::Ptr transformed_cloud(new SemanticCloud);
  transformPointCloud(*local_cloud_, *transformed_cloud, local_cloud_pose_);
  *global_cloud_ += *transformed_cloud;

  cout << "global cloud size: " << global_cloud_->size() << endl;
}

// get current pose
Matrix4f SimpleBirdseyeOdometer::get_current_pose() { return current_pose_; }

// get local semantic cloud
SemanticCloud::Ptr SimpleBirdseyeOdometer::get_local_cloud() {
  return local_cloud_;
}

// get global semantic cloud
SemanticCloud::Ptr SimpleBirdseyeOdometer::get_global_cloud(float voxel_size) {

  // downsampling
  pcl::VoxelGrid<SemanticPoint> sor;
  sor.setInputCloud(global_cloud_);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*global_cloud_);

  // return
  return global_cloud_;
}

// get trajectory
vector<Matrix4f> SimpleBirdseyeOdometer::get_trajectory() {
  return trajectory_;
}
}