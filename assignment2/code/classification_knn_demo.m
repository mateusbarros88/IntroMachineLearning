%% Classification with KNN K-Nearest Nabours


clear all;
close all;
clc;
datapath = '../../MNIST Dataset/';
addpath(genpath('../../Toolbox/'));
addpath(datapath);

% Digits to include in analysis (to include all, n = 0:9);
n = [0:9];
% Feature Mode [0:(pixels),1:(dont use),2:(1x272 v,h,radial histograms,
% radials in-out out-in profiles)].
mode = 2;
% cache feature data;
cache = 1; reset = 0; saveimgs = 0;
% rng(202322)) for report images.
rng(202322);
%% Load Data

classNames = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10'};
if ~cache || ~exist('data_cache_2.mat','file') || reset
    [X_train] = loadMNISTImages( ...
        [datapath 'train-images-idx3-ubyte/train-images.idx3-ubyte'] );
    [X_test, nrows, ncols] = loadMNISTImages( ...
        [datapath 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte'] );
    %X_train_ims = reshape(X_train,nrows,ncols,size(X_train,2));
       
    if mode ~= 0
        X_train = feature_extraction( X_train , nrows , ncols , mode )';
        X_test = feature_extraction( X_test , nrows , ncols , mode )';
        Y_train = loadMNISTLabels( ...
            [datapath 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'] );
        Y_test = loadMNISTLabels( ...
            [datapath 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'] );
    end
    if reset
        delete data_cache_2.mat;
    end
    if cache
        save('data_cache_2.mat','X_train','X_test','Y_test','Y_train','classNames');
    end
else
    load data_cache_2;
end

classLabels_train = classNames(Y_train+1);
classLabels_test = classNames(Y_test+1);

% Remove digits that are not to be inspected
j = ismember(Y_train, n);
X_train = X_train(j,:);
classLabels_train = classLabels_train(j);
classNames = classNames(n+1);
Y_train = cellfun(@(str) find(strcmp(str, classNames)), classLabels_train)-1;

j = ismember(Y_test, n);
X_test = X_test(j,:);
classLabels_test = classLabels_test(j);
Y_test = cellfun(@(str) find(strcmp(str, classNames)), classLabels_test)-1;
clear 'j'

%%
%% K-nearest neighbors
%roc = [];
for K = 2:5; % Number of neighbors

Distance = 'euclidean'; % Distance measure , cityblock, cosine, correlation

% Use knnclassify to find the K nearest neighbors
tic
y_test_est = knnclassify(X_test, X_train, Y_train, K, Distance);
toc

% Plot confusion matrix
mfig('Confusion matrix');
[acc, err] = confmatplot(classNames(Y_test+1), classNames(y_test_est+1));
title(sprintf('K=%d, Accuracy=%.1f%%, Error Rate=%.1f%%', K, ...
    acc, err));
roc(K,:) = [acc,err]
end
if saveimgs
    
    print('-depsc',sprintf('confusm_k%d',K));
    print('-djpeg',sprintf('confusm_k%d',K));
    
    %%copyfile('epsFig.eps','../../conf/img/var_explained.eps');
    %%print -djpeg epsFig
    %%copyfile('epsFig.jpg','../../conf/img/var_explained.jpg');
    %%delete('epsFig.eps');
    %%delete('epsFig.jpg');
end
%%
