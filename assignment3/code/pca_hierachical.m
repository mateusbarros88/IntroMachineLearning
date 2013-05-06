%% Load data, perform pca
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../Toolbox/02450Tools/');

load data_cache_2;

y = y_train;
X = X_train;


n = 0:9;

X = X(ismember(y,n),:);
y = y(ismember(y,n),:);

Y = bsxfun(@minus, X, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y, 'econ');

Z = U*S;

y = y(1:10000);
Z = Z(1:10000,:);

Z = Z(:,1:40);
N = length(Z);

%% Hierarchical clustering

% Maximum number of clusters
Maxclust = 50;

% Compute hierarchical clustering
L = linkage(Z, 'ward', 'euclidean');

% Compute clustering by thresholding the dendrogram
i = cluster(L, 'Maxclust', Maxclust);

%% Plot results

% Plot dendrogram
%mfig('Dendrogram'); clf;
%dendrogram(L);

% Plot data
mfig('Hierarchical'); clf; 
clusterplot(Z, y, i);
