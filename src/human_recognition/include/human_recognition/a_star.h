#ifndef _A_STAR_H_
#define _A_STAR_H_

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/filters/voxel_grid.h>
#include <geometry_msgs/Vector3.h>
#include <vector>

namespace human_recognition
{

#define min_value(a, b) \
    ({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _b : _a; })

class AStar
{
public:
  typedef struct Node{
    int x;
    int y;
    float distance;
    float weight;
    int visited;
  } Node;

public:
  AStar();

  void init_node(Node*, int, int, float);
  void update_neighbor(int, int, Node*);
  void set_cloud(pcl::PointCloud<pcl::PointXYZRGBA>);

  float findMinSumPath();
  float calc_dist(geometry_msgs::Vector3, geometry_msgs::Vector3);
  float run(geometry_msgs::Vector3, geometry_msgs::Vector3);

  Node* closest_unvisited();

private:
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;
  std::vector< std::vector<Node> > grid;
  int N;
};

}

#endif
