#include "NormalEstimator.h"
#include "NearestNeighbors.h"
#include "iostream"


// This method has to compute a normal per point in the 'points' vector and put it in the 
// 'normals' vector. The 'normals' vector already has the same size as the 'points' vector. 
// There is no need to push_back new elements, only to overwrite them ('normals[i] = ...;')
// In order to compute the normals you should use PCA. The provided 'NearestNeighbors' class
// wraps the nanoflann library that computes K-nearest neighbors effciently. 

void NormalEstimator::computePointCloudNormals(const vector<glm::vec3> &points, vector<glm::vec3> &normals)
{
    NearestNeighbors nn;
    nn.setPoints(&points);

    for (int i=0;i<points.size();i++)
    {
        std::vector<size_t> neighbors;
        std::vector<float>  dist_squared;
        glm::vec3 centroid;
		std::vector<glm::vec3> translatedPoints;

        int num_found = nn.getKNearestNeighbors(points[i],16,neighbors,dist_squared);

        for (int j=0; j<neighbors.size();j++)
        {
            centroid += points[neighbors[j]];
        }
        centroid /= neighbors.size();
		std::cout<<"The point is: ["<<points[i].x<<" "<<points[i].y<<" "<<centroid.z<<"]"<<std::endl;
        std::cout<<"The centroid is: ["<<centroid.x<<" "<<centroid.y<<" "<<centroid.z<<"]"<<std::endl;
		translatedPoints.push_back(points[i]-centroid);
		std::cout<<"The translated point is: ["<<translatedPoints[i].x<<" "<<translatedPoints[i].y<<" "<<translatedPoints[i].z<<"]"<<std::endl;


        
    }
}