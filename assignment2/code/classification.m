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

classNames = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10'};
classLabels = classNames(Labels+1);

clear ims nrows

%% Data specification

X = Data;
y = Labels;

[N M] = size(X);
attributeNames = int2str([1:M]');

clear Data Labels

%% Decision Tree
% Number of folds for crossvalidation
K = 5;

% Create holdout crossvalidation partition
CV = cvpartition(classNames(y+1), 'Kfold', K);

% Pruning levels
prune = 0:10;

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

    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train+1), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end
end

% Plot classification error
mfig('Digit decision tree: K-fold crossvalidatoin'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');