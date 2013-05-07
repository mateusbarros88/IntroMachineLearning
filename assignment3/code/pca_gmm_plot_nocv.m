%% Load data, perform pca
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../Toolbox/02450Tools/');

load data_cache_2;

n = 0:9;

y = y_train;
X = X_train;

% Limit
X = X(ismember(y,n),:);
y = y(ismember(y,n),:);

Y = bsxfun(@minus, X, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y, 'econ');

Z = U*S;

%Z = Z(1:20000,:);


%% Gaussian mixture model

% Number of clusters
K = length(n);

Z = Z(:,1:40);

% Fit model
G = gmdistribution.fit(Z, K, 'replicates', 10, 'regularize', 10e-6);

% Compute clustering
i = cluster(G, Z);

%% Plot results

% Plot clustering
mfig('GMM: Clustering'); clf;
clusterplot(Z, y, i);
title('Clustering of digits 0-9');

%% Print results

correct = nan(K,1);

for t = 1:K
    c = y(i==t);
    num = mode(c);
    
    correct(t) = sum(c == num);
    
    fprintf('Cluster %d has mode %d\n', t, num);
    fprintf('  Success rate: %d/%d (%.4f)\n', correct(t), length(c), correct(t)/length(c)*100);
end;

fprintf('\nOVERALL: %.4f\n', sum(correct)/length(i)*100);
