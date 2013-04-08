%% Load data and labels
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../MNIST Dataset/');

load data_cache_2

% This anonymous function takes as input a training and a test set. 
%  1. It fits a generalized linear model on the training set using glmfit.
%  2. It estimates the output of the test set using glmval.
%  3. It computes the sum of squared error.
funLinreg = @(X_train, y_train, X_test, y_test) ...
    sum((y_test-glmval(glmfit(X_train, y_train), ...
    X_test, 'identity')).^2);

%% Pick a number and attribute to predict
num = 4;
attr = 50;

% range = 1:56; % vert+hori
% range = 57:128; % radi
range = 129:200; % in-out
% range = 201:272; % out-in

% Extract only data matching the picked number
X = X_train(ismember(y_train, num), range); 
attributeNames = attributeNames(range);

% Extract the attribute we want to predict
y = X(:,attr);
X(:,attr) = [];
attributeNames(attr) = [];
[N,M] = size(X);

clear num attr

%% Setup cross-validation
K = 10;
CV = cvpartition(N, 'Kfold', K);

% Initialize variables
Features = nan(K,M);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_fs = nan(K,1);
Error_test_fs = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    % Use 10-fold crossvalidation for sequential feature selection
    tic
    [F, H] = sequentialfs(funLinreg, X_train, y_train);
    toc
    
    % Save the selected features
    Features(k,:) = F;
    
    % Compute squared error without using the input data at all
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);
    % Compute squared error without feature subset selection
    Error_train(k) = funLinreg(X_train, y_train, X_train, y_train);
    Error_test(k) = funLinreg(X_train, y_train, X_test, y_test);
    % Compute squared error with feature subset selection
    Error_train_fs(k) = funLinreg(X_train(:,F), y_train, X_train(:,F), y_train);
    Error_test_fs(k) = funLinreg(X_train(:,F), y_train, X_test(:,F), y_test);              
end

%% Display results
fprintf('\n');
fprintf('Linear regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
fprintf('Linear regression with sequential feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train_fs)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test_fs)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train_fs))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test_fs))/sum(Error_test_nofeatures));

%% Show the selected features
mfig('Attributes'); clf;
bmplot(1:K, attributeNames, Features);
xlabel('Attribute');
ylabel('Crossvalidation fold');
title('Attributes selected');

set(gca, 'XTick', 1, 'XTickLabel', '');
hx = get(gca,'XLabel'); % Handle to xlabel 
set(hx,'Units','data'); 
pos = get(hx,'Position'); 
y = pos(2);

% Place the new labels 
for i = 1:length(attributeNames),
    t(i) = text(i,y,attributeNames(i)); 
end;
set(t,'Rotation',90,'HorizontalAlignment','right') ;