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

Z = Z(:,1:20);
N = length(Z);

%% Gaussian mixture model

% Range of K's to try
%KRange = [10,20,30,40,50,60,70,80,90,100,110,120];
KRange = 45:55;
T = length(KRange);

% Allocate variables
BIC = nan(T,1);
AIC = nan(T,1);
CVE = zeros(T,1);

% Create crossvalidation partition for evaluation
CV = cvpartition(N, 'Kfold', 5);

% For each model order
for t = 1:T    
    % Get the current K
    K = KRange(t);
    
    % Display information
    fprintf('Fitting model for K=%d\n', K);
    
    % Fit model
    G = gmdistribution.fit(Z, K, 'regularize', 10e-6);
    
    % Get BIC and AIC
    BIC(t) = G.BIC;
    AIC(t) = G.AIC;
    
    % For each crossvalidation fold
    for k = 1:CV.NumTestSets
        fprintf('  Fold %d/%d\n', k, CV.NumTestSets);
        
        % Extract the training and test set
        Z_train = Z(CV.training(k), :);
        Z_test = Z(CV.test(k), :);
        
        % Fit model to training set
        G = gmdistribution.fit(Z_train, K, 'regularize', 10e-6);
        
        % Evaluation crossvalidation error
        [~, NLOGL] = posterior(G, Z_test);
        CVE(t) = CVE(t)+NLOGL;
    end
end


%% Plot results

mfig('GMM: Number of clusters'); clf; hold all
plot(KRange, BIC);
plot(KRange, AIC);
plot(KRange, 2*CVE);
legend('BIC', 'AIC', 'Crossvalidation');
xlabel('K');
