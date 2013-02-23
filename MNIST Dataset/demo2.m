
clear all;
close all;
clc;
load datamatfeatures.mat

Labels = loadMNISTLabels('train-labels-idx1-ubyte/train-labels.idx1-ubyte');
%classNames = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10'};
%classLabels = classNames(Labels+1);
%%
[Ztest nrows ncols] = loadMNISTImages('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte');
Ztest = feature_extraction(Ztest,nrows,ncols);
%%
ZLabels = loadMNISTLabels('t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte');
yest = knnclassify(Ztest', data', Labels);
errorRate = nnz(ZLabels~=yest)/length(ZLabels)
%% 
clear all;
close all;
clc;
fp = fopen('letter-recognition.data');
Data = zeros(20000,16);
Labels = zeros(20000,1);
for n = 1:20000
   Labels(n)= fscanf(fp,'%c',1);
   fscanf(fp,'%c',1);
   for c = 1:16
    Data(n,c) = fscanf(fp,'%d',1);
    fscanf(fp,'%c',1);
   end
   
end
fclose(fp);
yest = knnclassify(Data(16001:20000,:), Data(1:16000,:), Labels(1:16000));
errorRate = nnz(Labels(16001:20000)~=yest)/4000