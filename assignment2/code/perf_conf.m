%% Performance Comparison
load cv_error_results
confval =  2.7764;
k = 5;
%% KNN vs Decision Trees

d = error_knn - error_decision_tree';
mu = mean(d);
sigma = sqrt(var(d)/k);
dcv = [(mu - confval * sigma) , (mu + confval * sigma) ]
%SIGNIFICANT DIFFERENT.

%% KNN vs Naive Bayes


d = error_knn - error_naive_bayes;
mu = mean(d);
sigma = sqrt(var(d)/k);
dcv = [(mu - confval * sigma) , (mu + confval * sigma) ]
%SIGNIFICANT DIFFERENT.

%% NAIVE VS TREES

d = error_decision_tree' - error_naive_bayes;
mu = mean(d);
sigma = sqrt(var(d)/k);
dcv = [(mu - confval * sigma) , (mu + confval * sigma) ]


%% TEST

d = [0.1 0 0.1 0 0.3]
mu = mean(d);
sigma = sqrt(var(d)/k);
dcv = [(mu - confval * (sigma)) , (mu + confval * sigma) ]