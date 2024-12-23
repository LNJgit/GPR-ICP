#include "NormalEstimator.h"
#include "NearestNeighbors.h"
#include <Eigen/Dense>
#include "iostream"


// This method has to compute a normal per point in the 'points' vector and put it in the 
// 'normals' vector. The 'normals' vector already has the same size as the 'points' vector. 
// There is no need to push_back new elements, only to overwrite them ('normals[i] = ...;')
// In order to compute the normals you should use PCA. The provided 'NearestNeighbors' class
// wraps the nanoflann library that computes K-nearest neighbors effciently. 

void NormalEstimator::computePointCloudNormals(const vector<glm::vec3> &points, vector<glm::vec3> &normals)
{
    normals.resize(points.size());
    NearestNeighbors nn;
    nn.setPoints(&points);

    for (size_t i = 0; i < points.size(); i++)
    {
        std::vector<size_t> neighbors;
        std::vector<float> dist_squared;
        glm::vec3 centroid(0.0f, 0.0f, 0.0f);

        int num_found = nn.getKNearestNeighbors(points[i], 16, neighbors, dist_squared);
        if (num_found < 3) {
            // Not enough neighbors to compute a normal
            normals[i] = glm::vec3(0.0f, 0.0f, 0.0f);
            continue;
        }

        // Compute centroid
        for (size_t j = 0; j < neighbors.size(); j++)
        {
            centroid += points[neighbors[j]];
        }
        centroid /= static_cast<float>(neighbors.size());

        // Compute covariance matrix
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (size_t j = 0; j < neighbors.size(); j++)
        {
            glm::vec3 translatedPoint = points[neighbors[j]] - centroid;
            Eigen::Vector3d eigenTranslated(translatedPoint.x, translatedPoint.y, translatedPoint.z);
            covariance += eigenTranslated * eigenTranslated.transpose();
        }

        // Eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success) {
            normals[i] = glm::vec3(0.0f, 0.0f, 0.0f);
            continue;
        }

        Eigen::Vector3d eigenValues = solver.eigenvalues();
        Eigen::Matrix3d eigenVectors = solver.eigenvectors();

        // Extract normal (eigenvector corresponding to the smallest eigenvalue)
        int min_index;
        eigenValues.minCoeff(&min_index);
        Eigen::Vector3d normalEigenVector = eigenVectors.col(min_index);
        glm::vec3 normal(normalEigenVector[0], normalEigenVector[1], normalEigenVector[2]);

        // Normalize and orient normal
        normal = glm::normalize(normal);
        if (normal.z < 0)
        {
            normal = -normal;
        }

        normals[i] = normal;
    }
}
