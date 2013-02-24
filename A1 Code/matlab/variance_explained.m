%% Finds the Variance Explained by M-Principal Components.
clear all;
close all;
clc;
datapath = '../../MNIST Dataset/';
addpath(['../../Toolbox/MBox']);
addpath(['../../Toolbox/']);

% Digits to include in analysis (to include all, n = 0:9);
n = [0:9];
% The values to show variance explaination by
conf_val = [1 0.975 0.95 0.90 0.80 0.60];
% Feature Mode [0:(pixels),1:(dont use),2:(1x272 v,h,radial histograms,
% radials in-out out-in profiles)].
mode = 2;
% cache feature data;
cache = 1; reset = 0; saveimgs = 0;
% rng(sum('Report')) for report images.
rng(sum('Report Generation'));
%% Load Data
addpath(datapath);

if ~cache || ~exist('data_cache.mat','file') || reset
    [Data, nrows, ncols] = loadMNISTImages( ...
        [datapath 'train-images-idx3-ubyte/train-images.idx3-ubyte'] );
    
    if mode ~= 0
        Data = feature_extraction( Data , nrows , ncols , mode )';
    end
    if reset
        delete data_cache.mat;
    end
    if cache
        save('data_cache.mat','Data','nrows','nrows');
    end
else
    load data_cache;
end

Labels = loadMNISTLabels( ...
    [datapath 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'] );
classNames = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10'};
classLabels = classNames(Labels+1);

% Remove digits that are not to be inspected
j = ismember(Labels, n);
Data = Data(j,:);
classLabels = classLabels(j);
classNames = classNames(n+1);
Labels = cellfun(@(str) find(strcmp(str, classNames)), classLabels)-1;
clear 'j'
%% PCA
idx = 1:length(Labels);%find(Labels==1);
% Subtract the mean from the data
Y = bsxfun(@minus, Data, mean(Data));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y,'econ');


%% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);
rhosum = cumsum(rho);

close all
pcs = min(sum(repmat(rhosum,[1,length(conf_val)]) <= ...
    repmat(conf_val,[length(rhosum),1]))+1,numel(rhosum));

% Plot variance explained
figure1 =  mfig('Digits: Var. explained');  clf;
set(figure1,'DefaultTextInterpreter', 'latex')
axes1 = axes( 'Parent',figure1, ...
    'XTick',[pcs(end:-1:1)], ...
    'YTick',[rhosum(1) rhosum(pcs(end:-1:1))'], ...
    'DataAspectRatio',[140 1 1]);
set(figure1,'DefaultTextInterpreter', 'latex')
box(axes1,'on');
hold(axes1,'all');
plot(rhosum, 'Marker','.','Color',[0 0 1]);hold on

for j = 1:length(conf_val)
    npcs = [rhosum(1) pcs(j) pcs(j) ;rhosum(pcs(j)) rhosum(pcs(j)) rhosum(1)    ];
    plot(npcs(1,:),npcs(2,:),'r-');
end
ylim([rhosum(1) rhosum(end)]);

title('Variance explained by principal components','Interpreter','latex');
xlabel('M Principal component','Interpreter','latex');
ylabel('\% Variance explained by M PCs','Interpreter','latex');
axis tight

if saveimgs
    print -depsc epsFig
    copyfile('epsFig.eps','../../conf/img/var_explained.eps');
    print -djpeg epsFig
    copyfile('epsFig.jpg','../../conf/img/var_explained.jpg');
end

%% Corralation

figure1 =  mfig('Digits: Corralation');  clf;
set(figure1,'DefaultTextInterpreter', 'latex')

N = 1000;
ridx = randi([1,length(Data)],N,1);

hist(Labels(ridx),0:9);
[sortLabels sortIdx] = sort(Labels(ridx));
corrm = corr(Data(ridx,:)');
imagesc(corrm(sortIdx,sortIdx))%dont translate to get attribute corralation

figure1 =  mfig('Digits: Std');  clf;
set(figure1,'DefaultTextInterpreter', 'latex')
standalizedData = bsxfun(@minus, Data(ridx,:), mean(Data(ridx,:)));
standalizedData = bsxfun(@rdivide, standalizedData, std(Data(ridx,:)));
imagesc(max(min(standalizedData(sortIdx,:),3),-3));
colormap(hot);
%%

% Compute the projection onto the principal components
ridx = 1:length(U);
Z = U(ridx,:)*S;
co = [1 2];
mfig(['Digits: PCA']); clf; hold all; 
C = length(classNames);
for c = 0:C-1
    Xc = [ Z(Labels(ridx)==c,co(1)) Z(Labels(ridx)==c,co(2))];%  Z(Labels(ridx)==c,3)];
   % plot(Z(Labels(ridx)==c,1), Z(Labels(ridx)==c,2), 'o');
    error_ellipse(cov(Xc),'conf',0.75,'mu',mean(Xc),'style','-');
end
legend(classNames);
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('PCA of digits data');

%%
%S = 60;M = 1;N =10; C((S-M*N+N-1),(N-1)) / C(S+N-1,(N-1))