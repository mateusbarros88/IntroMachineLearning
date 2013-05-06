%% Load data, perform pca
clear all;
close all;
clc;
addpath('../../Toolbox/MBox');
addpath('../../Toolbox/');
addpath('../../Toolbox/02450Tools/');
datapath = '../../MNIST Dataset/';
load data_cache_2;
addpath(datapath);
    [Data, nrows, ncols] = loadMNISTImages( ...
        [datapath 'train-images-idx3-ubyte/train-images.idx3-ubyte'] );
    ims = reshape(Data,nrows,ncols,size(Data,2));   
%%
n = 0:9;
n = 5; %Remove this to generate them all

y = y_train;
X = X_train;

for i = n 
    % Limit
    Xt = X(y==i,:);
    allidx = find(y==i);
    yt = y(allidx);

    Y = bsxfun(@minus, Xt, mean(Xt));

    % Obtain the PCA solution by calculate the SVD of Y
    [U, S, V] = svd(Y, 'econ');

    Z = U*S;

    str = sprintf('PCA on digit %d', i);
    
    mfig(str); clf;
    plot(Z(:,1), Z(:,2), 'o');
    xlabel('PC 1');
    ylabel('PC 2');
    title(str);
end;
%%
if length(n) ~= 1
    return;
end

% SET n equal one number and run it all again.
figure1 = figure; %mfig(['Digits: Projections']); clf; hold all; 
set(figure1,'DefaultTextInterpreter', 'latex');
rng(202322);
Z = U*S;
N=length(Z);
idx = zeros(N,1);
scale = 6;
mapsize=500*scale;
map = zeros(mapsize,mapsize,3);
cmap =[1 0 0    %0
       0 1 0    %1
       0.5 0 1  %2
       1 1 0    %3
       0 1 1    %4
       1 0.5 1  %5
       0.5 0.5 0%6
       .2 0.5 0.5%7
       0.5 1 .5 %8
       0.5 0 0 ]%9
maxx = -inf;
minx = inf;
maxy = -inf;
miny = inf;

idx = randi([1,length(allidx)],N,1);

Xc = [ Z(idx,1) Z(idx,2)]*scale*9;
    for n = 1:length(Xc);
       % (round(Xc(n,2))-14:round(Xc(n,2))+13)+1000
       % (round(Xc(n,1))-14:round(Xc(n,1))+13)+1000
       try
           
       if ~any( map(  (round(Xc(n,2))-14:round(Xc(n,2))+13)+mapsize/2, (round(Xc(n,1))-14:round(Xc(n,1))+13)+mapsize/2,: ))
        im = repmat(flipud(ims(:,:,allidx(idx(n))))/255,[1 1 3]);
        im(:,:,1) = im(:,:,1)*cmap(y(allidx(idx(n)))+1,1);
        im(:,:,2) = im(:,:,2)*cmap(y(allidx(idx(n)))+1,2);
        im(:,:,3) = im(:,:,3)*cmap(y(allidx(idx(n)))+1,3);
        if miny > round(Xc(n,2))-14
            miny = round(Xc(n,2))-14;
        end
        if minx > round(Xc(n,1))-14
            minx = round(Xc(n,1))-14;
        end
        if maxx < round(Xc(n,1))+13
            maxx = round(Xc(n,1))+13;
        end
        if maxy < round(Xc(n,2))+13
            maxy = round(Xc(n,2))+13;
        end
           map(  (round(Xc(n,2))-14:round(Xc(n,2))+13)+mapsize/2, ... 
               (round(Xc(n,1))-14:round(Xc(n,1))+13)+mapsize/2,: ) = ...
               im;
        %imagesc([Xc(n,1)-14.5 Xc(n,1)+14.5],[Xc(n,2)-14.5 Xc(n,2)+14.5], repmat(ims(:,:,idx(n))/255,[1 1 3]));
       end
       catch
           
       end
    end
    image(1-map);

 %   xlim([minx+100 maxx-300]+mapsize/2);
 %   ylim([miny+400 maxy-100]+mapsize/2);
axis equal
%set(gca,'XTickLabel',(get(gca,'XTick')-mapsize/2) / scale/9 ); 
%set(gca,'YTickLabel',(get(gca,'YTick')-mapsize/2) / scale/9); 
xlabel('PC1');
ylabel('PC2');
title('Projections of data to PC1 and PC2')