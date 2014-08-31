function [W,Dist] = compute_relation(data,scale_sig,order)
%
%      [W,distances] = compute_relation(data,scale_sig) 
%       Input: data= Feature_dimension x Num_data
%       ouput: W = pair-wise data similarity matrix
%              Dist = pair-wise Euclidean distance
%
%
% Jianbo Shi, 1997 

[~,numstocks]=size(data);


Dist = zeros(numstocks,numstocks);
for j = 1:length(numstocks),
  Dist(j,:) = (sqrt((data(1,:)-data(1,j)).^2 +...
                (data(2,:)-data(2,j)).^2));
end

% distances = X2distances(data');



if (~exist('scale_sig')),
    %scale_sig = 0.05*max(distances(:));
    sorted_distances = sort(Dist);
    vec = sorted_distances(2,:);
    scale_sig = mean(vec);
end

if (~exist('order')),
  order = 2;
end

%tmp = (distances/scale_sig).^order;
tmp = Dist.^2/(2*scale_sig*scale_sig);
W = exp(-tmp);




