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

for i = n 
    % Limit
    Xt = X(y==i,:);
    yt = y(y==i,:);

    Y = bsxfun(@minus, Xt, mean(Xt));

    % Obtain the PCA solution by calculate the SVD of Y
    [U, S, V] = svd(Y, 'econ');

    Z = U*S;

    str = sprintf('PCA on digit %d', i);
    
    mfig(str); clf;
    plot(Z(:,1), Z(:,2), 'o');
    xlabel('PC 1');
    ylabel('PC 2');
    title(str);
end;
