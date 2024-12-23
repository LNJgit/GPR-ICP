#include <iostream>
#include <algorithm>
#include "IterativeClosestPoint.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <glm/gtc/matrix_transform.hpp>


void IterativeClosestPoint::setClouds(PointCloud *pointCloud1, PointCloud *pointCloud2)
{
	cloud1 = pointCloud1;
	cloud2 = pointCloud2;
	knn.setPoints(&(cloud1->getPoints()));
}

// This method should mark the border points in cloud 1. It also changes their color (for example to red).
// You will need to add an attribute to this class that stores this property for all points in cloud 1. 

void IterativeClosestPoint::markBorderPoints()
{
    std::vector<size_t> neighbor_indices;
    std::vector<float> distances;
    std::vector<glm::vec3> points = cloud1->getPoints();
	std::vector<glm::vec4>& colors = cloud1->getColors(); // GET REFERENCE TO COLORS VECTOR
    borderPoints.resize(points.size(), false); // RESIZE TO MATCH POINT COUNT

    for (size_t i = 0; i < points.size(); i++) {
        // STEP 1: FIND K-NEAREST NEIGHBORS
        knn.getKNearestNeighbors(points[i], 16, neighbor_indices, distances);

        // IF THERE ARE NOT ENOUGH NEIGHBORS, MARK AS BORDER POINT
        if (neighbor_indices.size() < 3) {
            borderPoints[i] = true;
			colors[i] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // SET TO RED
            continue;
        }

        // STEP 2: COMPUTE CENTROID OF NEIGHBORS
        glm::vec3 centroid(0.0f, 0.0f, 0.0f);
        for (size_t j = 0; j < neighbor_indices.size(); j++) {
            centroid += points[neighbor_indices[j]];
        }
        centroid /= static_cast<float>(neighbor_indices.size());

        // STEP 3: COMPUTE COVARIANCE MATRIX
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (size_t j = 0; j < neighbor_indices.size(); j++) {
            glm::vec3 diff = points[neighbor_indices[j]] - centroid;
            Eigen::Vector3f diff_eigen(diff.x, diff.y, diff.z);
            covariance += diff_eigen * diff_eigen.transpose();
        }

        // STEP 4: PERFORM PCA
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Eigen::Vector3f eigenvalues = solver.eigenvalues();
        Eigen::Matrix3f eigenvectors = solver.eigenvectors();

        // STEP 5: PROJECT ONTO XY-PLANE
        Eigen::Vector3f normal = eigenvectors.col(0); // SMALLEST EIGENVECTOR
        std::vector<float> angles;
        for (size_t j = 0; j < neighbor_indices.size(); j++) {
            glm::vec3 diff = points[neighbor_indices[j]] - points[i];
            Eigen::Vector3f diff_proj = Eigen::Vector3f(diff.x, diff.y, diff.z) - 
                                        (normal.dot(Eigen::Vector3f(diff.x, diff.y, diff.z)) * normal);
            float angle = atan2(diff_proj.y(), diff_proj.x());
            angles.push_back(angle);
        }

        // STEP 6: SORT ANGLES AND FIND MAX GAP
        std::sort(angles.begin(), angles.end());
        float maxDeltaAlpha = 0.0f;
        for (size_t j = 1; j < angles.size(); j++) {
            maxDeltaAlpha = std::max(maxDeltaAlpha, angles[j] - angles[j - 1]);
        }
        maxDeltaAlpha = std::max(maxDeltaAlpha, static_cast<float>(2 * M_PI - (angles.back() - angles.front())));

        // STEP 7: MARK AS BORDER POINT IF MAX GAP EXCEEDS THRESHOLD
        if (maxDeltaAlpha > M_PI / 2) { // THRESHOLD CAN BE ADJUSTED
            borderPoints[i] = true;
            colors[i] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // SET TO RED
        }
    }
}




// This method should compute the closest point in cloud 1 for all non border points in cloud 2. 
// This correspondence will be useful to compute the ICP step matrix that will get cloud 2 closer to cloud 1.
// Store the correspondence in this class as the following method is going to need it.
// As it is evident in its signature this method also returns the correspondence. The application draws this if available.

std::vector<int> *IterativeClosestPoint::computeCorrespondence()
{
    // RESIZE THE CORRESPONDENCES VECTOR
    correspondences.resize(cloud2->getPoints().size(), -1);

    // GET THE POINTS FROM BOTH CLOUDS
    const std::vector<glm::vec3>& points1 = cloud1->getPoints();
    const std::vector<glm::vec3>& points2 = cloud2->getPoints();

    // SETUP THE KD-TREE WITH CLOUD1 POINTS
    knn.setPoints(&points1); // ASSUMES `knn` IS A KD-TREE IMPLEMENTATION

    // LOOP THROUGH ALL NON-BORDER POINTS IN CLOUD2
    for (size_t i = 0; i < points2.size(); i++) {
        // FIND THE CLOSEST POINT IN CLOUD1
        std::vector<size_t> neighbor_indices;
        std::vector<float> distances;
        knn.getKNearestNeighbors(points2[i], 1, neighbor_indices, distances); // FIND CLOSEST NEIGHBOR

        // STORE THE CLOSEST POINT INDEX
        if (!neighbor_indices.empty()) {
            correspondences[i] = neighbor_indices[0];
        }
    }

    // RETURN A POINTER TO THE CORRESPONDENCES VECTOR
    return &correspondences;
}





// This method should compute the rotation and translation of an ICP step from the correspondence
// information between clouds 1 and 2. Both should be encoded in the returned 4x4 matrix.
// To do this use the SVD algorithm in Eigen.

glm::mat4 IterativeClosestPoint::computeICPStep()
{
	// EXTRACT CORRESPONDING POINTS
    const std::vector<glm::vec3>& points1 = cloud1->getPoints();
    const std::vector<glm::vec3>& points2 = cloud2->getPoints();
    std::vector<glm::vec3> P, Q;

    for (size_t i = 0; i < correspondences.size(); i++) {
        if (correspondences[i] != -1) {
            Q.push_back(points1[correspondences[i]]);
            P.push_back(points2[i]);
        }
    }

	// COMPUTE CENTROIDS 
	glm::vec3 centroidP(0.0f);
	glm::vec3 centroidQ(0.0f);
	for (size_t i = 0; i<<P.size(); i++)
	{
		centroidP +=P[i];
		centroidQ +=Q[i];
	}
	centroidP /= static_cast<float>(P.size());
    centroidQ /= static_cast<float>(Q.size());

	// COMPUTE COVARIANCE MATRIX
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < P.size(); i++) {
        Eigen::Vector3f p(P[i].x - centroidP.x, P[i].y - centroidP.y, P[i].z - centroidP.z);
        Eigen::Vector3f q(Q[i].x - centroidQ.x, Q[i].y - centroidQ.y, Q[i].z - centroidQ.z);
        S += q * p.transpose();
    }

	// PERFORM SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

	// COMPUTE ROTATION MATRIX
    Eigen::Matrix3f R = V * U.transpose();
    if (R.determinant() < 0) {
        V.col(2) *= -1; // HANDLE REFLECTION CASE
        R = V * U.transpose();
    }

	// COMPUTE TRANSLATION VECTOR
    Eigen::Vector3f t = Eigen::Vector3f(centroidP.x, centroidP.y, centroidP.z) - 
                        R * Eigen::Vector3f(centroidQ.x, centroidQ.y, centroidQ.z);

    // CONSTRUCT 4x4 TRANSFORMATION MATRIX
    glm::mat4 transform = glm::mat4(1.0f);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            transform[i][j] = R(i, j);
        }
        transform[i][3] = t(i);
    }

    return transform;
}


// This method should perform the whole ICP algorithm with as many steps as needed.
// It should stop when maxSteps are performed, when the Frobenius norm of the transformation matrix of
// a step is smaller than a small threshold, or when the correspondence does not change from the 
// previous step.
std::vector<int> *IterativeClosestPoint::computeFullICP(unsigned int maxSteps)
{
    // INITIALIZE THRESHOLDS FOR CONVERGENCE
    const float translationThreshold = 1e-6; // THRESHOLD FOR TRANSLATION CONVERGENCE
    const float rotationThreshold = 1e-6;   // THRESHOLD FOR ROTATION CONVERGENCE
    glm::mat4 cumulativeTransform = glm::mat4(1.0f); // START WITH IDENTITY TRANSFORM

    std::vector<int> previousCorrespondences; // TO TRACK CORRESPONDENCE CHANGES

    for (unsigned int step = 0; step < maxSteps; ++step) {
        std::cout << "ICP Step: " << step + 1 << std::endl;

        // STEP 1: COMPUTE CORRESPONDENCES
        computeCorrespondence();

        // STEP 2: CHECK IF CORRESPONDENCES HAVE CHANGED (CONVERGENCE CRITERIA)
        if (step > 0 && correspondences == previousCorrespondences) {
            std::cout << "CONVERGENCE: CORRESPONDENCES HAVE NOT CHANGED." << std::endl;
            break;
        }
        previousCorrespondences = correspondences; // UPDATE PREVIOUS CORRESPONDENCES

        // STEP 3: COMPUTE TRANSFORMATION (ROTATION AND TRANSLATION)
        glm::mat4 transform = computeICPStep();

        // DEBUG: PRINT TRANSFORMATION MATRIX
        std::cout << "TRANSFORMATION MATRIX:\n";
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << transform[i][j] << " ";
            }
            std::cout << std::endl;
        }

        // STEP 4: CHECK TRANSFORMATION CONVERGENCE (ROTATION AND TRANSLATION)
        glm::vec3 translation(transform[3][0], transform[3][1], transform[3][2]); // EXTRACT TRANSLATION VECTOR
        glm::mat3 rotation = glm::mat3(transform); // EXTRACT ROTATION MATRIX
        glm::mat3 identity(1.0f); // IDENTITY MATRIX FOR ROTATION COMPARISON
        // COMPUTE THE DIFFERENCE BETWEEN THE ROTATION MATRIX AND THE IDENTITY MATRIX
		glm::mat3 diff = rotation - identity;

		// CALCULATE THE FROBENIUS NORM
		float rotationError = 0.0f;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				rotationError += diff[i][j] * diff[i][j];
			}
		}
		rotationError = std::sqrt(rotationError); // TAKE THE SQUARE ROOT
        float translationError = glm::length(translation); // TRANSLATION ERROR

        if (rotationError < rotationThreshold && translationError < translationThreshold) {
            std::cout << "CONVERGENCE: TRANSFORMATION BELOW THRESHOLDS." << std::endl;
            break;
        }

        // STEP 5: APPLY TRANSFORMATION TO CLOUD2
        cloud2->applyTransformation(transform);

        // STEP 6: UPDATE CUMULATIVE TRANSFORMATION
        cumulativeTransform = transform * cumulativeTransform;
    }

    // RETURN THE FINAL CORRESPONDENCES
    return &correspondences;
}






