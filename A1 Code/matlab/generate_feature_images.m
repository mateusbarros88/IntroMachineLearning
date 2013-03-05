%% Finds the Variance Explained by M-Principal Components.
clear all;
close all;
clc;
datapath = '../../MNIST Dataset/';
addpath(['../../Toolbox/MBox']);
addpath(['../../Toolbox/']);
addpath(datapath);

saveimgs = 1;
inversecolor = 1;
%%

[Data, nrows, ncols] = loadMNISTImages( ...
        [datapath 'train-images-idx3-ubyte/train-images.idx3-ubyte'] );
ims = reshape(Data,nrows,ncols,size(Data,2));   
Data = Data';

Labels = loadMNISTLabels( ...
    [datapath 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'] );
classNames = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10'};
classLabels = classNames(Labels+1);
    %%
close all
N = 10;
fullimg = ones(nrows*N,ncols*N)*255;
for l = 0:9
    imsc = ims(:,:,find(Labels==l,N));
    for n = 1:N   
        fullimg(((n-1)*nrows+1):((n-1)*nrows+1+nrows-1),((l)*ncols+1):((l)*ncols+1+ncols-1)) = imsc(:,:,n);

    end
end
figure1 =  mfig('Digits: Corralation');  clf;
set(figure1,'DefaultTextInterpreter', 'latex')

imagesc((255*inversecolor)-fullimg*(1-(2*inversecolor))), colormap(gray(256))

set(gca,'XTick', 14.5:28:ncols*10);
set(gca,'XTickLabel',classNames); 
set(gca,'YTick', 14.5:28:nrows*N);
set(gca,'YTickLabel',1:N);
xlabel('Classes')
ylabel('Example n')
title('The n=1 .. 2 .. N first examples of each class');
%set(gca,'DataAspectRatio',[1 1 1 ]);
axis image

if saveimgs
    print -depsc epsFig
    copyfile('epsFig.eps','../../conf/img/image_examples.eps');
    print -djpeg epsFig
    copyfile('epsFig.jpg','../../conf/img/image_examples.jpg');
    delete('epsFig.eps');
    delete('epsFig.jpg');
end

%%
%% Correlation

figure1 =  mfig('Digits: Corralation');  clf;
set(figure1,'DefaultTextInterpreter', 'latex')
corrm = corr(Data);
imagesc(corrm)%dont translate to get attribute corralation

%%
n = hist(sortLabels);
nn = cumsum(n);

set(gca,'XTick', nn-n/2);
set(gca,'XTickLabel',classNames); 
set(gca,'YTick', nn-n/2);
set(gca,'YTickLabel',classNames);
%set(gca,'DataAspectRatio',[1 1 1 ]);
axis equal square
xlim([0 N])
cb = colorbar('peer',gca);
ylabel(cb, 'Corralation Cooficient')
title('The corralation of 2000 samples sorted by class');
xlabel('2000 Samples over 10 classes');
ylabel('2000 Samples over 10 classes');


%%
figure1 =  mfig('Digits: Features');  clf;
set(figure1,'DefaultTextInterpreter', 'latex')


N = 10;
fullimg = ones(nrows*N*2,(272+1+2)*3)*255;
L1 = [0 1 2; 3 6 9];
for row = 1:2
    L=L1(row,:);
for l = 0:2;
    imsc = ims(:,:,find(Labels==L(l+1),N));
    Data1 = feature_extraction( Data(find(Labels==L(l+1),N),:)' , nrows , ncols , 2 )';
    for n = 1:N  
        Z = ones(28,size(Data1,2)+3)*255;
        Z(1:28,2:29) = imsc(:,:,n);
        for z = 1:size(Data1,2)
           Z(:,30+z) = [zeros(1,28-Data1(n,z)) ones(1,Data1(n,z))*255]'; 
        end
       fullimg(((n-1)*nrows+1)+((row-1)*nrows*N):((n-1)*nrows+1+nrows-1)+((row-1)*nrows*N),((l)*302+1):((l)*302+1+302-1)) = Z;
         %  image(Z);
        %   pause

    end
end
end


imagesc((255*inversecolor)-fullimg*(1-(2*inversecolor))), colormap(gray(256))

set(gca,'XTick', 14.5:28:ncols*10);
set(gca,'XTickLabel',''); 
set(gca,'YTick', 14.5:28:nrows*N*2);
set(gca,'YTickLabel',[1:N,1:N]);
xlabel('Classes and features')
ylabel('Example n')
title('The n=1 .. 2 .. N first examples of some class');
%set(gca,'DataAspectRatio',[1 1 1 ]);
axis image

if saveimgs
    print -depsc epsFig
    copyfile('epsFig.eps','../../conf/img/feature_examples369.eps');
    print -djpeg epsFig
    copyfile('epsFig.jpg','../../conf/img/feature_examples369.jpg');
    delete('epsFig.eps');
    delete('epsFig.jpg');
end

%%
subplot(1,2,1)
image(ims(:,:,1)),colormap gray
subplot(1,2,2)

bar(Data1);
%%
clc
B1 = ims(:,:,find(Labels==0,1));
 B = Data(find(Labels==0,1),:)';
 A = feature_extraction( B , nrows , ncols , 2 );
 
T = zeros(1,28*2);
T(28) = 1;

Z = [];
 for n = 0:27
     Z(n+1,:) = repmat(T(28-n:28+27-n),1,28);
 end
 for n = 0:27
     Z(28+n+1,:) = [zeros(1,28*n) ones(1,28) zeros(1,784-28*(n+1))];
 end

 
 (B>128)'*Z'%[ones(1,28) zeros(1,784-28); zeros(1,28*7) ones(1,28) zeros(1,784-28*8) ]'
 A(1:28*2)'

 (Z*(B>128))' ==  A(1:28*2)'
 
 %imagesc(reshape((Z \ A(1:28*2))',28,28))
 sum(abs((B>128)' - (Z \ A(1:28*2))'))
 imagesc(reshape((B>128),28,28))
