#ifndef _ITERATIVE_CLOSEST_POINT_INCLUDE
#define _ITERATIVE_CLOSEST_POINT_INCLUDE


#include "PointCloud.h"
#include "NearestNeighbors.h"


class IterativeClosestPoint
{

public:
	void setClouds(PointCloud *pointCloud1, PointCloud *pointCloud2);
	
	void markBorderPoints();
	vector<int> *computeCorrespondence();
	glm::mat4 computeICPStep();
	
	vector<int> *computeFullICP(unsigned int maxSteps = 200);
	
private:
	PointCloud *cloud1, *cloud2;
	NearestNeighbors knn;
    std::vector<bool> borderPoints; // STORES WHETHER EACH POINT IS A BORDER POINT
	std::vector<int> correspondences; //STORES THE INDICES OF THE CLOSEST POINTS OF CLOUD 1 
};


#endif // _ITERATIVE_CLOSEST_POINT_INCLUDE


