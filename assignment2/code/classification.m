%% Load data and labels
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../MNIST Dataset/');

load data_cache_2

classLabels = classNames(y_train+1);

%% Data specification

X = X_train;
y = y_train;
[N,M] = size(X);

%% Decision Tree
load cv_split;
% CV = cvpartition(classNames(y+1), 'Kfold', K);
K = CV.NumTestSets;

% Pruning levels
prune = 25:35;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    tic
    % Fit classification tree to training set
    T{k} = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames, ...
        'prune', 'on', ...
        'minparent', 10);
    toc
    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train+1), eval(T{k}, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test+1), eval(T{k}, X_test, prune(n))));
    end
    toc
end

%% Plot classification error
mfig('Digit decision tree: K-fold crossvalidatoin'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');