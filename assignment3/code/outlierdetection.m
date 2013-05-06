%% outlier

clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../Toolbox/02450Tools/');
datapath = '../../MNIST Dataset/';
load data_cache_2;
addpath(datapath);
   [Data, nrows, ncols] = loadMNISTImages( ...
        [datapath 'train-images-idx3-ubyte/train-images.idx3-ubyte'] );
    ims = reshape(Data,nrows,ncols,size(Data,2));   
%%

n = 5;

y = y_train;
X = X_train;

Xt = X(y==n,:);
allidx = find(y==n);
yt = y(allidx,:);
%%
% exercise 11.2.4

% Neighbor to use
K = 5;

% Find the k nearest neighbors
[i, D] = knnsearch(Xt, Xt, 'K', K+1);

% Outlier score
f = D(:,K+1);

% Sort the outlier scores
[y,i] = sort(f, 'descend');
%%
% Display the index of the lowest density data object
% The outlier should have index 1001
disp(allidx(i(1:5)));

% Plot kernel density estimate outlier scores
mfig('Distance: Outlier score'); clf;
bar(y(1:200));







%%
   % exercise 11.1.1

% Number of data objects
N = size(X,1);

% x-values to evaluate the histogram
x = linspace(-10, 10, 50)';

% Number of attributes
M = size(X,2);

% Allocate variable for data
X = nan(N,M);

% Mean and covariances
m = [1 3 6];
s = [1 .5 2];

% For each data object
for n = 1:N
    k = discreternd([1/3 1/3 1/3]);    
    X(n,1) = normrnd(m(k), sqrt(s(k)));
end

% Plot histogram
mfig('Histogram'); clf;
hist(X, x);
    
