%% Load data, perform pca
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../Toolbox/02450Tools/');

load data_cache_2;

n = [1,0];

y = y_train;
X = X_train;

% Limit
X = X(ismember(y,n),:);
y = y(ismember(y,n),:);

Y = bsxfun(@minus, X, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y, 'econ');

Z = U*S;


%% Gaussian mixture model

% Number of clusters
K = 2;

Z = Z(:,1:5);

% Fit model
G = gmdistribution.fit(Z, K, 'regularize', 10e-9);

% Compute clustering
i = cluster(G, Z);

%% Extract cluster centers
X_c = G.mu;
Sigma_c=G.Sigma;

%% Plot results

% Plot clustering
mfig('GMM: Clustering'); clf;
clusterplot(Z, y, i);

