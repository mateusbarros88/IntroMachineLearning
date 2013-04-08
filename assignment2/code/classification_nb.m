%% Load data
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../MNIST Dataset/');

load data_cache_2

classLabels = classNames(y_train+1);

X = X_train;
y = y_train;

%% K-fold crossvalidation
% K = 10;
% CV = cvpartition(y, 'Kfold', K);
load cv_split
K = CV.NumTestSets;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'uniform';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_val = X(CV.test(k), :);
    y_val = y(CV.test(k));

    % Fit naive Bayes classifier to training set
    tic
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data
    y_val_est = predict(NB, X_val);
    toc
    % Compute error rate
    Error(k) = sum(y_val~=y_val_est); % Count the number of errors
end

%% Print the error rate
%fprintf('Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);
fprintf('Error rate: %.3f\n', Error./CV.TestSize');