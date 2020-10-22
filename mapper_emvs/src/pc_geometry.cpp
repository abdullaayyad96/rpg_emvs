#include <mapper_emvs/pc_geometry.hpp>
#include <mapper_emvs/mapper_emvs.hpp>
#include <mapper_emvs/median_filtering.hpp>
#include <pcl/filters/radius_outlier_removal.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>

#include "Mission_Management/my_msg.h"
#include <vector>

namespace EMVS {

using namespace geometry_utils;


void PCGeometry::FitPlanetoPC(PointCloud::Ptr cloud_filtered, pcl::ModelCoefficients::Ptr coefficients)
{
  //TODO: Project points back on to the plane
  //pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  
  // Create the segmentation object
  pcl::SACSegmentation<PointType> seg;
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);
  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  }

  Eigen::Vector4f plane_parameters;
  pcl::ModelCoefficients plane_coeff; 
  plane_coeff.values.resize (4);
  plane_coeff.values[0] =  coefficients->values[0]; 
  plane_coeff.values[1] =  coefficients->values[1]; 
  plane_coeff.values[2] =  coefficients->values[2]; 
  plane_coeff.values[3] =  coefficients->values[3]; 

  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;

}

void PCGeometry::PlaneRotationVector(pcl::ModelCoefficients::Ptr coefficients, geometry_utils::Transformation last_pose, Eigen::Vector4f& Quat)
{
  Eigen::Vector3f NormaltoPlane;
  NormaltoPlane[0] = coefficients->values[0]; //x-component
  NormaltoPlane[1] = coefficients->values[1]; //y-component
  NormaltoPlane[2] = coefficients->values[2]; //z-component
  
  float ax = atan2(sqrt(pow(NormaltoPlane[1],2)+pow(NormaltoPlane[2],2)),NormaltoPlane[0]);
  float ay = atan2(sqrt(pow(NormaltoPlane[2],2)+pow(NormaltoPlane[0],2)),NormaltoPlane[1]);
  
  Eigen::Vector3f RotationVector;
  Eigen::Vector3f X(1,0,0);
  Eigen::Vector3f Y(0,1,0);
  Eigen::Vector3f Z(0,0,1);
  RotationVector = Z.cross(NormaltoPlane); //TODO: check all rotations

  float RotAngle = 0;
  if (RotationVector.norm() > std::numeric_limits<float>::epsilon())
  {
    RotationVector = RotationVector/RotationVector.norm();  
    RotAngle = acos(NormaltoPlane.dot(Z)/NormaltoPlane.norm());
  }
  else
  {
    RotAngle = 0;
    RotationVector << 0, 0, 0;
  }  
  
  // Find Quaternion of the roatation vector
  Quat[0] = cos(RotAngle/2);
  Quat[1] = RotationVector[0] * sin(RotAngle/2);
  Quat[2] = RotationVector[1] * sin(RotAngle/2);
  Quat[3] = RotationVector[2] * sin(RotAngle/2);

  LOG(INFO) << "Rot Angle :" << RotAngle*(180/3.14);
  LOG(INFO) << "Rot Vector :" << RotationVector[0] << ", " << RotationVector[1] << ", " << RotationVector[2];
  
  LOG(INFO) << "Quat X Y Z W :" << Quat[1] << ", " << Quat[2] << ", " << Quat[3] << ", " << Quat[0];
  
}

void PCGeometry::ProjectPointsOnPlane(pcl::ModelCoefficients::Ptr coefficients, PointCloud::Ptr cloud_filtered, PointCloud::Ptr& holes_pos)
{ 
  Eigen::Vector3f NormaltoPlane;
  NormaltoPlane[0] = coefficients->values[0]; //x-component
  NormaltoPlane[1] = coefficients->values[1]; //y-component
  NormaltoPlane[2] = coefficients->values[2]; //z-component

  Eigen::Vector3f DispVect;

  Eigen::Vector3f PointOnPlane;
  PointOnPlane[0] = coefficients->values[0];
  PointOnPlane[1] = coefficients->values[1];
  PointOnPlane[2] = -coefficients->values[3] - (std::pow(PointOnPlane[0],2) + std::pow(PointOnPlane[1],2)) / coefficients->values[2]; //TODO: zero divisiin check
  
  Eigen::Vector3f ParallelToNormal;
  Eigen::Vector3f PerpendicularToNormal;
  
  for(int i=0; i < cloud_filtered->size(); i++)
  { 
    DispVect[0] = cloud_filtered->points[i].x - PointOnPlane[0];
    DispVect[1] = cloud_filtered->points[i].y - PointOnPlane[1];
    DispVect[2] = cloud_filtered->points[i].z - PointOnPlane[2];
    ParallelToNormal = ((DispVect.dot(NormaltoPlane))/std::pow(NormaltoPlane.norm(),2)) * NormaltoPlane;
    PerpendicularToNormal = DispVect - ParallelToNormal;
  
    EMVS::PointType P;
    P.x = PointOnPlane[0] + PerpendicularToNormal[0];
    P.y = PointOnPlane[1] + PerpendicularToNormal[1];
    P.z = PointOnPlane[2] + PerpendicularToNormal[2];
    holes_pos->push_back(P);
  }
  LOG(INFO) << "Projected Points: " << holes_pos->points[0] << holes_pos->points[1] << "Before Projection: " << cloud_filtered->points[0] << cloud_filtered->points[1];

}

void PCGeometry::PlaneinInertial(PointCloud::Ptr holes_pos, geometry_utils::Transformation last_pose, Eigen::Vector4f Quat, Eigen::Vector4f& PlaneQuatInertial, geometry_msgs::Point& point, int i)
{
  //Get Camera pose
  kindr::minimal::RotationQuaternion CamPose = last_pose.getRotation();

  //Transform from camera frame to inertial frame //TODO: check all rotations
  // PlaneQuatInertial[0] = Quat[0] * CamPose.w() - Quat[1] * CamPose.x() - Quat[2] * CamPose.y() - Quat[3] * CamPose.z();  // 1
  // PlaneQuatInertial[1] = Quat[0] * CamPose.x() + Quat[1] * CamPose.w() + Quat[2] * CamPose.z() - Quat[3] * CamPose.y();  // i
  // PlaneQuatInertial[2] = Quat[0] * CamPose.y() - Quat[1] * CamPose.z() + Quat[2] * CamPose.w() + Quat[3] * CamPose.x();  // j
  // PlaneQuatInertial[3] = Quat[0] * CamPose.z() + Quat[1] * CamPose.y() - Quat[2] * CamPose.x() + Quat[3] * CamPose.w();  // k

  Eigen::Vector4f CamPoseQuat;
  CamPoseQuat <<  CamPose.w(), CamPose.x(), CamPose.y(), CamPose.z();
  //Transform from camera frame to inertial frame 
  PlaneQuatInertial[0] = CamPoseQuat[0] * Quat[0] - CamPoseQuat[1] * Quat[1] - CamPoseQuat[2] * Quat[2] - CamPoseQuat[3] * Quat[3];  // 1
  PlaneQuatInertial[1] = CamPoseQuat[0] * Quat[1] + CamPoseQuat[1] * Quat[0] + CamPoseQuat[2] * Quat[3] - CamPoseQuat[3] * Quat[2];  // i
  PlaneQuatInertial[2] = CamPoseQuat[0] * Quat[2] - CamPoseQuat[1] * Quat[3] + CamPoseQuat[2] * Quat[0] + CamPoseQuat[3] * Quat[1];  // j
  PlaneQuatInertial[3] = CamPoseQuat[0] * Quat[3] + CamPoseQuat[1] * Quat[2] - CamPoseQuat[2] * Quat[1] + CamPoseQuat[3] * Quat[0];  // k

  Eigen::Matrix4d TransformationMat = last_pose.getTransformationMatrix();

  Eigen::Vector4d pcinCamFrame;
  Eigen::Vector4d pcinInertialFrame;


  pcinCamFrame[0] = holes_pos->points[i].x;
  pcinCamFrame[1] = holes_pos->points[i].y;
  pcinCamFrame[2] = holes_pos->points[i].z;
  pcinCamFrame[3] = 1.0;
  pcinInertialFrame = TransformationMat * pcinCamFrame;
  point.x = pcinInertialFrame[0];
  point.y = pcinInertialFrame[1];
  point.z = pcinInertialFrame[2];


   pcl::PCDWriter writer;
   writer.write ("Holes.pcd", *holes_pos, false);
  //NavigatetoPlane(pcinInertialFrame, PlaneQuatInertial);
}

// void PCGeometry::NavigatetoPlane(Eigen::Vector4d pc_, Eigen::Vector4f PlaneQuatInertial)
// {
//   geometry_msgs::Pose Pose;
//   Pose.position.x = pc_[0];
//   Pose.position.y = pc_[1];
//   Pose.position.z = pc_[2] + 0.1;
//   Pose.orientation.x = PlaneQuatInertial[1];
//   Pose.orientation.y = PlaneQuatInertial[2];
//   Pose.orientation.z = PlaneQuatInertial[3];
//   Pose.orientation.w = PlaneQuatInertial[0];
//   LOG(INFO) << " Pose message :" << Pose.orientation.x << Pose.orientation.y << Pose.orientation.z << Pose.orientation.w;
//   this->cmd_pos_pub.publish(Pose);
// }

}
