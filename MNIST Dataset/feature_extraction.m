function [Data] = feature_extraction(D, nrows, ncols, mode)
% Feature Extration

% Read the contents back into an array
%[D, nrows, ncols] = loadMNISTImages('train-images-idx3-ubyte/train-images.idx3-ubyte');
if nargin <4
    mode = 0;
end
attributes = [16 28*2+72*3];
Data = zeros(attributes(mode),size(D,2));

for n= 1:size(Data,2)
    switch mode
        case 1
            Data(:,n) = extract1(D(:,n),nrows,ncols);
        case 2
            Data(:,n) = extract2(D(:,n),nrows,ncols)';
    end
end

if mode == 4
    max_val= max(Data,[],2);
    min_val= min(Data,[],2);
    Data = Data - (min_val*ones(1,size(Data,2)));
    Data = floor(Data ./ ((max_val - min_val)*ones(1,size(Data,2))) * 16);
elseif mode == 3
    max_val= max(Data,[],2);
min_val= min(Data,[],2);
Data = Data - (min_val*ones(1,size(Data,2)));
Data = floor(Data ./ ((max_val - min_val)*ones(1,size(Data,2))) * 255);
end

end

function feat = extract2(column_vec,rows,cols)
im = reshape(column_vec,rows,cols)>128;
vprofile = sum(im);
hprofile = sum(im,2)';

k = 0:71;
phi = 5*k;
rr = fix((14.5- ((1:14)'*sind(phi))) -14)+14;
cc = fix((14.5 -((1:14)'*cosd(phi))) -14)+14;
idx = (cc-1)*28+rr;
rprofile = sum(im(idx));

%[dist sidx] = sort(sqrt(px.^2+py.^2));
pradial = zeros(2,length(k));

save('test.mat','im','idx','rr','cc');
%pause
for n = k+1
   % sidx(:,n)
    id = idx(:,n);
    
    a = find(im(id)==1,1,'first');
    b = find(im(id)==1,1,'last');
    %[n a b]
    %pause
    
    if(isempty(a))
        a = 0;
        b = 0;
    end

    pradial(1,n) =a;
    pradial(2,n) =b;
end
%pradial
%subplot(2,2,1)
%imagesc(im);hold on;axis equal;
%for k = [0 90 180 270]/5


%plot(fix((14.5- (pradial(1,k+1)'*cosd(k*5))) -14)+14,fix((14.5- (pradial(1,k+1)'*sind(k*5))) -14)+14,'ro');
%plot(fix((14.5- (pradial(2,k+1)'*cosd(k*5))) -14)+14,fix((14.5- (pradial(2,k+1)'*sind(k*5))) -14)+14,'yo');

%end

%subplot(2,2,2)
%bar(hprofile)
%subplot(2,2,3)
%bar(vprofile)
%subplot(2,2,4)
%bar(pradial(:)')


%pause
feat = [hprofile vprofile rprofile pradial(:)'];

end
function feat = extract1(column_vec,rows,cols)

im = reshape(column_vec,rows,cols)>128;
%subplot(1,2,1);
%imagesc(im);colormap(gray(256));
feat=zeros(16,1);

stats = regionprops(im,'BoundingBox');
bb = stats.BoundingBox;
feat(1) = bb(1)+bb(3)/2; %The horizontal position,  counting pixels from the left edge of the image,  of the center 
                      %of the smallest rectangular box that can be drawn with all  "on" pixels inside the box
feat(2) = bb(2) +bb(4)/2;%The  vertical  position,  counting  pixels  from the  bottom,  of the  above box
feat(3) = bb(3);
feat(4) = bb(4);
feat(5) = sum(sum(im));

[X,Y] = meshgrid(1:rows,1:cols);
X = X-cols/2 +0.5;
Y = Y-rows/2 +0.5;
feat(6) = mean(mean(X(im)));
feat(7) = mean(mean(Y(im)));


Xn = X.^2;
Yn = Y.^2;
feat(8) = mean(mean(X(im)));
feat(9) = mean(mean(Y(im)));

feat(10) = mean(mean(X(im).*Y(im)));

feat(11) = mean(mean(Xn(im).*Y(im)));
feat(12) = mean(mean(Yn(im).*X(im)));
edgemap = false(rows,cols);
for r = 1:rows
    edgemap(r,1) = im(r,1);
    for c=1:cols-1
        edgemap(r,c+1) = ~im(r,c) && im(r,c+1);
    end
end
feat(13) = mean(mean(Y(edgemap)));
feat(14) = sum(sum(Y(edgemap)));
edgemap = false(rows,cols);
for c = 1:cols
    edgemap(1,c) = im(1,c);
    for r=1:rows-1
        edgemap(r+1,c) = im(r,c) && ~im(r+1,c);
    end
end
feat(15) = mean(mean(X(edgemap)));
feat(16) = sum(sum(X(edgemap)));
%subplot(1,2,2);
%imagesc(edgemap);

end
