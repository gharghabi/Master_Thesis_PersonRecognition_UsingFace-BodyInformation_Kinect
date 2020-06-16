#include <human_recognition/a_star.h>

namespace human_recognition
{

AStar::AStar() {}

void AStar::init_node(Node* n, int x, int y, float w)
{
    n->x = x;
    n->y = y;
    n->distance = 1000000;
    n->weight = w;
    n->visited = false;
}

float AStar::run(geometry_msgs::Vector3 positionFirst, geometry_msgs::Vector3 positionSecont)
{
    std::vector<geometry_msgs::Vector3> pointInArea;
    float minx, miny, minz, maxx, maxy, maxz;
    if (positionFirst.x > positionSecont.x) {
        maxx = positionFirst.x;
        minx = positionSecont.x;
    } else {
        minx = positionFirst.x;
        maxx = positionSecont.x;
    }

    if (positionFirst.y>positionSecont.y) {
        maxy = positionFirst.y;
        miny = positionSecont.y;
    } else {
        miny = positionFirst.y;
        maxy = positionSecont.y;
    }

    pointInArea.push_back(positionFirst);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudVoxeld (new pcl::PointCloud<pcl::PointXYZRGBA>);

    for (int l=0; l<cloud.size(); ++l) {
        if(abs(cloud.at(l).x-minx)<abs(maxx-minx) && abs(cloud.at(l).y-miny)<abs(maxy-miny)) {
            cloudVoxeld->push_back(cloud.at(l));
        }
    }

    float voxelsize = 0.025f;
    while (cloudVoxeld->size() > 100) {
        pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        sor.setInputCloud (cloudVoxeld);
        sor.setLeafSize (voxelsize, voxelsize, voxelsize);
        sor.filter (*cloudVoxeld);
        voxelsize += 0.005;
    }

    for (int l=0; l<cloudVoxeld->size(); ++l) {
        geometry_msgs::Vector3 positionRow;
        positionRow.x = cloudVoxeld->at(l).x;
        positionRow.y = cloudVoxeld->at(l).y;
        positionRow.z = cloudVoxeld->at(l).z;
        pointInArea.push_back(positionRow);
    }
    pointInArea.push_back(positionSecont);

    N = pointInArea.size();
    grid.resize(N);
    for (int i=0 ; i < N ; ++i) {
        grid[i].resize(N);
    }
    for (int f=0; f<N; ++f) {
        for (int ff=0; ff<N; ++ff) {
            float weight = calc_dist(pointInArea.at(f),pointInArea.at(ff));
            init_node(&grid[f][ff],f,ff,weight);
        }
    }

    float minSum = findMinSumPath();
    return minSum;
}

void AStar::update_neighbor(int dx, int dy, Node* u)
{
    int newy = (u->y) + dy;
    int newx = (u->x) + dx;
    if (newx >= 0 && newy >= 0 && newx<N  && newy < N){
        Node* n = &grid[newx][newy];
        if (!n->visited){
            n->distance = min_value(n->distance,u->distance + n->weight);
        }
    }
}

void AStar::set_cloud(pcl::PointCloud<pcl::PointXYZRGBA> cloud)
{
    this->cloud = cloud;
}

AStar::Node* AStar::closest_unvisited()
{
    int i, j;
    float min = 1000000;
    Node* minNode;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            if (! grid[i][j].visited) {
                if (grid[i][j].distance < min) {
                    min = grid[i][j].distance;
                    minNode = &grid[i][j];
                }
            }
        }
    }
    return minNode;
}

float AStar::findMinSumPath()
{
    Node* u;
    grid[0][0].distance = grid[0][0].weight;
    int i;
    while (! grid[N-1][N-1].visited) {
        u = closest_unvisited();
        u->visited = true;
        update_neighbor(0,-1,u);
        update_neighbor(0,1,u);
        update_neighbor(1,0,u);
        update_neighbor(-1,0,u);

    }
    return grid[N-1][N-1].distance;
}

float AStar::calc_dist(geometry_msgs::Vector3 positionFirst, geometry_msgs::Vector3 positionSecont)
{
    return (abs(positionFirst.x-positionSecont.x)+ abs(positionFirst.y-positionSecont.y)+abs(positionFirst.z-positionSecont.z));
}

}
