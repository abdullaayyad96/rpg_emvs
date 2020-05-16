#include <mapper_emvs/data_loading.hpp>
#include <mapper_emvs/mapper_emvs.hpp>

#include <image_geometry/pinhole_camera_model.h>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/io/pcd_io.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>

// Input parameters
DEFINE_string(bag_filename, "input.bag", "Path to the rosbag");
DEFINE_string(event_topic, "/dvs/events", "Name of the event topic (default: /dvs/events)");
DEFINE_string(pose_topic, "/optitrack/davis", "Name of the pose topic (default: /optitrack/davis)");
DEFINE_string(camera_info_topic, "/dvs/camera_info", "Name of the camera info topic (default: /dvs/camera_info)");
DEFINE_double(start_time_s, 0.0, "Start time in seconds (default: 0.0)");
DEFINE_double(stop_time_s, 1000.0, "Stop time in seconds (default: 1000.0)");

// Disparity Space Image (DSI) parameters. Section 5.2 in the IJCV paper.
DEFINE_int32(dimX, 0, "X dimension of the voxel grid (if 0, will use the X dim of the event camera) (default: 0)");
DEFINE_int32(dimY, 0, "Y dimension of the voxel grid (if 0, will use the Y dim of the event camera) (default: 0)");
DEFINE_int32(dimZ, 100, "Z dimension of the voxel grid (default: 100) must be <= 256");
DEFINE_double(fov_deg, 0.0, "Field of view of the DSI, in degrees (if < 10, will use the FoV of the event camera) (default: 0.0)");
DEFINE_double(min_depth, 0.3, "Min depth, in meters (default: 0.3)");
DEFINE_double(max_depth, 5.0, "Max depth, in meters (default: 5.0)");

// Depth map parameters (selection and noise removal). Section 5.2.3 in the IJCV paper.
DEFINE_int32(adaptive_threshold_kernel_size, 5, "Size of the Gaussian kernel used for adaptive thresholding. (default: 5)");
DEFINE_double(adaptive_threshold_c, 5., "A value in [0, 255]. The smaller the noisier and more dense reconstruction (default: 5.)");
DEFINE_int32(median_filter_size, 5, "Size of the median filter used to clean the depth map. (default: 5)");

// Point cloud parameters (noise removal). Section 5.2.4 in the IJCV paper.
DEFINE_double(radius_search, 0.05, "Size of the radius filter. (default: 0.05)");
DEFINE_int32(min_num_neighbors, 3, "Minimum number of points for the radius filter. (default: 3)");

class DepthEstimator
{
  public:
    DepthEstimator(const std::string& camero_info_topic, const std::string& event_topic, const std::string& pose_topic, float distance_thresh,
                   float dim_X, float dim_Y, float dim_Z, float min_depth, float max_depth, float fov_deg,
                   float adaprive_threshold_kernel_size, float adaptive_threshold_c, float median_filter_size)
    {
      event_subs_ = ros_node_.subscribe(event_topic, 10, &DepthEstimator::EventsCallback, this);
      cam_info_subs_ = ros_node_.subscribe(camero_info_topic, 10, &DepthEstimator::CamInfoCallback, this);
      pose_sub = ros_node_.subscribe(pose_topic, 10, &DepthEstimator::PoseCallback, this);

      dsi_shape_ = EMVS::ShapeDSI(dim_X, dim_Y, dim_Z,
                                min_depth, max_depth,
                                fov_deg);

      opts_depth_map_.adaptive_threshold_kernel_size_ = adaprive_threshold_kernel_size;
      opts_depth_map_.adaptive_threshold_c_ = adaptive_threshold_c;
      opts_depth_map_.median_filter_size_ = median_filter_size;

      update_distance_ = distance_thresh;
    }

  private:
      ros::NodeHandle ros_node_;

      std::vector<dvs_msgs::Event> events_;
      image_geometry::PinholeCameraModel cam_;

      bool cam_initialized = false;
      bool pose_initialized_ = false;

      float update_distance_;

      std::map<ros::Time, geometry_utils::Transformation> poses_;
      std::vector<geometry_msgs::PoseStamped> pose_list_;
      std::vector<ros::Time> pose_timestaps_;

      geometry_msgs::PoseStamped processed_pose_;

      ros::Publisher depthmap_publisher_ = ros_node_.advertise<sensor_msgs::Image>("depth_image", 3);
    
      ros::Subscriber event_subs_; 
      ros::Subscriber cam_info_subs_;
      ros::Subscriber pose_sub;

      LinearTrajectory trajectory_;

      EMVS::ShapeDSI dsi_shape_;
      EMVS::MapperEMVS mapper_;
      cv::Mat depth_map_, confidence_map_, semidense_mask_, depth_map_255_;
      EMVS::OptionsDepthMap opts_depth_map_;
      sensor_msgs::Image ros_depth_map_;
      cv_bridge::CvImage depth_map_bridge_;

      void EventsCallback(const dvs_msgs::EventArray::ConstPtr &event_stream)
      {
        ROS_INFO("Event Stream Received");
        for (uint i = 0 ; i < event_stream->events.size() ; i++)
        {
          this->events_.push_back(event_stream->events[i]);
        }

      }

      void CamInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &camera_info)
      {
        ROS_INFO("Camera Info Received");
        this->cam_.fromCameraInfo(*camera_info);
        this->cam_initialized = true;

        //initialize mapper
        new(&this->mapper_) EMVS::MapperEMVS(this->cam_, this->dsi_shape_);

        //Disable camera info callbacks
        this->cam_info_subs_.shutdown();        
      }

      void PoseCallback(const geometry_msgs::PoseStamped::ConstPtr &camera_pose)
      {
        ROS_INFO("Pose Received");
        if (!pose_initialized_)
        {
          this->processed_pose_ = *camera_pose;
          pose_initialized_ = true;
        }

        this->pose_list_.push_back(*camera_pose);
        this->pose_timestaps_.push_back(camera_pose->header.stamp);

        const Eigen::Vector3d position(camera_pose->pose.position.x,
                                      camera_pose->pose.position.y,
                                      camera_pose->pose.position.z);
        const Eigen::Quaterniond quat(camera_pose->pose.orientation.w,
                                      camera_pose->pose.orientation.x,
                                      camera_pose->pose.orientation.y,
                                      camera_pose->pose.orientation.z);

        geometry_utils::Transformation T(position, quat);
        poses_.insert(std::pair<ros::Time, geometry_utils::Transformation>(camera_pose->header.stamp, T));

        double distance_to_last_keyframe = std::sqrt( std::pow(camera_pose->pose.position.x - this->processed_pose_.pose.position.x, 2) +
                                                      std::pow(camera_pose->pose.position.y - this->processed_pose_.pose.position.y, 2) +
                                                      std::pow(camera_pose->pose.position.z - this->processed_pose_.pose.position.z, 2));

        if (this->processed_pose_.header.stamp.toSec() > camera_pose->header.stamp.toSec())
        {
          this->processed_pose_ = *camera_pose;
          this->poses_.clear();
          this->pose_list_.clear();
          this->pose_timestaps_.clear();
          this->events_.clear();
        }
        else if ((distance_to_last_keyframe > update_distance_) && this->cam_initialized && (this->pose_list_.size() > 2))
        {
          this->ProcessEvents();
          this->processed_pose_ = *camera_pose;
          poses_.insert(std::pair<ros::Time, geometry_utils::Transformation>(camera_pose->header.stamp, T));
        }
      }

      void ProcessEvents()
      {
        ROS_INFO("Processing Events ...");
        this->trajectory_ = LinearTrajectory(this->poses_);
        
        // Set the position of the reference view in the middle of the trajectory
        geometry_utils::Transformation T1_;
        ros::Time t1_;
        this->trajectory_.getLastControlPose(&T1_, &t1_);

        ROS_INFO("Evaluating DSI ...");
        //evaluate DSI
        this->mapper_.evaluateDSI(this->events_, this->trajectory_ , T1_.inverse());

        // Write the DSI (3D voxel grid) to disk
        this->mapper_.dsi_.writeGridNpy("dsi.npy");
  
        //evaluate depth map
        this->mapper_.getDepthMapFromDSI(this->depth_map_, this->confidence_map_, this->semidense_mask_, this->opts_depth_map_);
        this->depth_map_255_ = (this->depth_map_ - this->dsi_shape_.min_depth_) * (255.0 / (this->dsi_shape_.max_depth_ - this->dsi_shape_.min_depth_));
        cv::imwrite("depth_map.png", this->depth_map_255_);

        ROS_INFO("Publishing Depth Map ...");
        //publish depth map
        this->ros_depth_map_.header.seq = 1; // user defined counter
        this->ros_depth_map_.header.stamp = ros::Time::now(); // time
        //this->img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, this->corner_heatmap_cv_mono8);
        this->depth_map_bridge_ = cv_bridge::CvImage(this->ros_depth_map_.header, sensor_msgs::image_encodings::TYPE_8SC1, this->depth_map_255_);
        
        this->depth_map_bridge_.toImageMsg(this->ros_depth_map_); // from cv_bridge to sensor_msgs::Image

        this->depthmap_publisher_.publish(this->ros_depth_map_);

        this->events_.clear();
        this->poses_.clear();   
        this->pose_list_.clear();
        this->pose_timestaps_.clear();
      }
    
};

/*
 * Load a set of events and poses from a rosbag,
 * compute the disparity space image (DSI),
 * extract a depth map (and point cloud) from the DSI,
 * and save to disk.
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_estimation");
	ros::NodeHandle nh_;

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  DepthEstimator depth_estimator(FLAGS_camera_info_topic, FLAGS_event_topic, FLAGS_pose_topic, 0.2,
                                FLAGS_dimX, FLAGS_dimY, FLAGS_dimZ, FLAGS_min_depth, FLAGS_max_depth, FLAGS_fov_deg,
                                FLAGS_adaptive_threshold_kernel_size, FLAGS_adaptive_threshold_c, FLAGS_median_filter_size);

  ros::spin();


  return 0;
}