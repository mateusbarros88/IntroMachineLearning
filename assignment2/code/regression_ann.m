%% Load data
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../MNIST Dataset/');

load linear_regression_attr50 'Features' 'X' 'y' 'N' 'attributeNames'
Features = logical(Features);

[~,ind] = min(sum(Features,2));
toKeep = Features(ind,:);
%toKeep = find(sum(Features)==max(sum(Features))); % Common features only

X = X(:,toKeep);
attributeNames = attributeNames(:,toKeep);

%% K-fold crossvalidation
K = 5;
CV = cvpartition(N, 'Kfold', K);

% Parameters for neural network classifier
NHiddenUnits = 10;  % Number of hidden units
NTrain = 1; % Number of re-trains of neural network

% Variable for classification error
Error = nan(K,1);
bestnet = cell(K,1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    % Fit neural network to training set
    tic
    MSEBest = inf;
    for t = 1:NTrain
        netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits);
        if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; end
    end
    toc
    
    % Predict model on test data
    y_test_est = round(bestnet{k}.t_pred_test);    
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

%% Print the error rate
fprintf('Error rate: %.3f\n', sum(Error)./sum(CV.TestSize));
