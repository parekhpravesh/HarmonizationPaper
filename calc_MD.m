function MD = calc_MD(matrix)
% Function to calculate Mahalanobis Distance from a reference distribution
%% Input:
% matrix:       three dimensional matrix where rows are samples, columns
%               are features, and the third dimension indexes the sites
%% Output:
% MD:           calculated Mahalanobis distance of each site from the
%               reference distribution [sites x 1]
%% Notes:
% First, a reference distribution is created by calculating the average of
% average of features across all sites (grand mean) and the pooled
% covariance matrix (which is the average covariance matrix across sites)
% Next, for each site, we calculate the Mahalanobis distance from this
% reference distribution
% 
%% Authors:
% Bhalerao, Gaurav
% Parekh, Pravesh
% December 31, 2021
% ADBS

% Find number of sites and number of features
numSites    = size(matrix,3);
numFeatures = size(matrix,2);

% Initialize
allMeans    = zeros(numFeatures, numSites);
tmpCov      = zeros(numFeatures, numFeatures);

for sites = 1:numSites

    % Get means for every site and every feature
    allMeans(:, sites) = mean(matrix(:,:,sites));
    
    % Sum of all covariances
    tmpCov = tmpCov + cov(matrix(:,:,sites));
end

% Grand mean and pooled covariance
overallCov     = tmpCov./numSites;
overallMean    = mean(allMeans,2);

% MD from the reference (grand mean and covariance)
MD = zeros(numSites,1);
for sites = 1:numSites
    term1       = (allMeans(:,sites) - overallMean);
    % MD(sites,1) = sqrt(term1 * inv(overallCov) * term1);
    MD(sites,1) = sqrt(term1' * (overallCov \ term1));
end