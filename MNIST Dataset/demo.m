%% Demo
clear all;
close all;
clc;

%mex load_data.c;

% Digits to include in analysis (to include all, n = 1:10);
n = [0:9];

% Number of principal components for reconstruction
K = 12;

% Digits to visualize
nD = 1:5;

% The values to show variance explaination by
conf_val = [0.975 0.95 0.90 0.80 0.60];

% Feature Mode
mode = 2;

%%
% Read the contents back into an array
[Data, nrows, ncols] = loadMNISTImages('train-images-idx3-ubyte/train-images.idx3-ubyte');
if mode ~= 0
    Data = feature_extraction(Data,28,28,mode);
end
%load datamatfeatures.mat
%Data = data;
%clear data;
%center of mass
%[rr,cc] = meshgrid(1:nrows,1:ncols);
%Mt = sum(Data);
%c1 = round(sum(Data .* repmat(rr(:),[1,size(Data,2)])) ./ Mt);
%c2 = round(sum(Data .* repmat(cc(:),[1,size(Data,2)])) ./ Mt);



Labels = loadMNISTLabels('train-labels-idx1-ubyte/train-labels.idx1-ubyte');
classNames = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10'};
classLabels = classNames(Labels+1);

% Remove digits that are not to be inspected
j = ismember(Labels, n);
Data = Data(:,j);
classLabels = classLabels(j);
classNames = classNames(n+1);
Labels = cellfun(@(str) find(strcmp(str, classNames)), classLabels)-1;
clear 'j'

%% PCA
idx = 1:length(Labels);%find(Labels==1);
% Subtract the mean from the data
Y = bsxfun(@minus, Data, mean(Data,2))';

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y,'econ');

% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);
rhosum = cumsum(rho);
%%
close all
clear npcs
pcs = sum(repmat(rhosum,[1,length(conf_val)]) < ...
    repmat(conf_val,[length(rhosum),1]))+1


% Plot variance explained
mfig('Digits: Var. explained');  clf;
plot(rhosum(1:max(pcs)), 'o-');hold on

for j = 1:5
    npcs = [0 pcs(j) pcs(j) ;rhosum(pcs(j)) rhosum(pcs(j)) 0]
plot(npcs(1,:),npcs(2,:),'r-')
end
title('Variance explained by principal components');
xlabel('Number of Principal component');
ylabel('Variance % explained by N PCs');


%% Plot PCA of data

% Compute the projection onto the principal components
Z = U*S;
combo = [1 2
    1 3
    2 3];

for co = 1:length(combo)
    
mfig(['Digits: PCA' co+'0']); clf; hold all; 
C = length(classNames);
for c = 0:C-1
    plot(Z(Labels==c,combo(co,1)), Z(Labels==c,combo(co,2)), 'o');
end
legend(classNames);
xlabel(['PC ' combo(co,1)+'0']);
ylabel(['PC ' combo(co,2)+'0']);
title('PCA of digits data');
colormap(hot)
hold off
end
%%

%groups(Labels==0) = '1';
for pcn = [1:5]
    
mfig(['Digits: PC' num2str(pcn)  ' Spread']); clf;
boxplot(Z(:,pcn),classLabels)

end
%mean(Z(Labels==1,1))
%std(Z(Labels==1,1))
%%

mfig(['Digits: Reconstruction']); clf;
for n = 0:9;
for K=1:length(pcs);

subplot(10,5,5*n+K)
W = Z(:,1:pcs(K))*V(:,1:pcs(K))';
I = reshape(mean(W(find(Labels==n,1,'last'),:)+repmat(mean(Data,2)',[1,1]),1),[28,28]);
imagesc(I);
drawnow;
colormap(hot)
end

end
%subplot(1,2,2)
%imagesc(reshape(Data(:,n),[28,28]));
%colormap(hot);
%% Test Classifier

[Ytest, nrows, ncols] = loadMNISTImages('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte');
if mode ~= 0
    Ytest = feature_extraction(Ytest,28,28,mode);
end

ZLabels = loadMNISTLabels('t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte');
Ytest = bsxfun(@minus, Ytest, mean(Data,2))';



yest = knnclassify(Ytest,Y, Labels);
errorRate1 = nnz(ZLabels~=yest)/length(ZLabels)
%%
errorRate2 = [];
k=1:5:50;
for K=k
    
Z = Y*V(:,1:K);
Ztest = Ytest*V(:,1:K);
yest = knnclassify(Ztest,Z, Labels);
errorRate2 = [errorRate2 nnz(ZLabels~=yest)/length(ZLabels)];

end


