#include <iostream>
//#include <Eigen/Geometry> 
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int main()
{
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Fill in the cloud data
  cloud.width    = 15;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.resize (cloud.width * cloud.height);

  Eigen::RowVectorXd X(15);
  Eigen::RowVectorXd Y(15);
  Eigen::RowVectorXd Z(15);

  X << -20,-20, -20, -10, -10, -10,  0, 0,   0, 10, 10,  10, 20, 20,  20;
  Y <<  10, 0 , -10,  10,   0, -10, 10, 0, -10, 10,  0, -10, 10,  0, -10;
  Z <<  0 , 0 ,   0,   0,   0,   0,  0, 0,   0,  0,  0,   0,  0,  0,   0;
  
  int i=0;
  for (auto& point: cloud)
  {
    point.x = X[i]/100;
    point.y = Y[i]/100;
    point.z = Z[i];
    i++;
  }

  pcl::io::savePCDFileASCII ("plate_pcd.pcd", cloud);
  std::cerr << "Saved " << cloud.size () << " data points to test_pcd.pcd." << std::endl;

  for (const auto& point: cloud)
  {
    std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;
  }
  return (0);
}