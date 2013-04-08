%% Load data and labels
clear all;
close all;
clc;
datapath = '../../MNIST Dataset/';
addpath(['../../Toolbox/MBox']);
addpath(['../../Toolbox/']);
addpath(datapath);

load data_cache

Labels = loadMNISTLabels( ...
    [datapath 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'] );

% This anonymous function takes as input a training and a test set. 
%  1. It fits a generalized linear model on the training set using glmfit.
%  2. It estimates the output of the test set using glmval.
%  3. It computes the sum of squared error.
funLinreg = @(X_train, y_train, X_test, y_test) ...
    sum((y_test-glmval(glmfit(X_train, y_train), ...
    X_test, 'identity')).^2);

clear ims nrows

%% Pick a number and attribute to predict
num = 4;
attr = 50;

% Extract only data matching the picked number
% X = Data(ismember(Labels, num),1:56); % Vert+hori
% X = Data(ismember(Labels, num),57:128); % radials
X = Data(ismember(Labels, num),129:200); % In-out
% X = Data(ismember(Labels, num),201:272); % Out-in

% Combine attributes
% [N,M] = size(X);
% c = 4;
% combinedData = nan(N, M/c);
% for i=1:M/c,
%     range = 1+(i-1)*c:i*c;
%     combinedData(:,i) = sum(X(:,range),2);
% end;
% X = combinedData;


% Extract the attribute we want to predict
y = X(:,attr);
X(:,attr) = [];
[N,M] = size(X);

attributeNames = 1:length(X(1,:));

clear Labels num attr i c range

%% Setup cross-validation
K = 3;
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
    [F, H] = sequentialfs(funLinreg, X_train, y_train);
    
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

% Show the selected features
mfig('Attributes'); clf;
bmplot(attributeNames, 1:K, Features');
xlabel('Crossvalidation fold');
ylabel('Attribute');
title('Attributes selected');
