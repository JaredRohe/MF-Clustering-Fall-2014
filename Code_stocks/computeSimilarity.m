function [W,Rank,Dist,logreturns] = computeSimilarity(slidingwindow,neighbors,weight,metric)
%% Info

load('CRSPSP500dates07_12.mat'); % make sure correct set of dates

[M,N]=size(slidingwindow);

stockcount=N;


%% Raw price data to logarthmic returns

% Percent Change
pchange = zeros(M-1,N);
for i=1:N
    for j=2:M
     pchange(j-1,i)=(slidingwindow(j,i)-slidingwindow(j-1,i))/slidingwindow(j-1,i);
    end
end

%Log Returns
logreturns=zeros(M-1,N);
pchange=1+pchange;

for i=1:N
    logreturns(:,i)=log(pchange(:,i));
end

%% Standard Normalization
% Rank=zeros(M-1,N);
% for i =1:N
%     Rank(:,i)=(logreturns(:,i)-mean(logreturns(:,i)))./std(logreturns(:,i));
% end

%% cumulative distribution function

Rank=zeros(M-1,N);

for i=1:N
    for j=1:M-1
        x=logreturns(j,i);
        numlessthan=logreturns(:,i)<=x;
        thresh=sum(numlessthan);
        Rank(j,i)=thresh/(M-1);
    end
end


%% Computing Distance and Similarity
% % my distance
% if strcmp(metric,'returns')
%     data=logreturns;
% else
%     data=Rank;
% end
% 
% Dist=zeros(stockcount,stockcount);
% 
% for i=1:N
%     for j=i+1:N
%         v1=data(:,i);
%         v2=data(:,j);
%         
%         EDist= sqrt(sum((v2 - v1).^2));
%         
%         Dist(i,j)=EDist;
%         Dist(j,i)=EDist;
%         
%     end
% end

% original distance

if strcmp(metric,'returns')
    data=logreturns;
else
    data=Rank;
end


[W,Dist] = compute_relation(data);


%% Sparsifiy


[M,N]=size(W);

allidx=[1:N];

k=neighbors;

if k~=0
    for i=1:M

            [~,idx]=sort(W(i,:),'descend');
            maxidx=idx(1,2:k+1);
            zerothese=setdiff(allidx,maxidx);

            W(i,zerothese)=0;



    end





    %Make symmetric

    asymmatrix=W-W';
    for i=1:M
    asymindices=find(asymmatrix(i,:)~=0);
        for j=asymindices
            if asymmatrix(i,j)<0
                W(i,j)=W(j,i);
            else
                W(j,i)=W(i,j);

            end

        end


    end
    
    %Make unweighted
    if weight==0
        indices=find(W~=0);
        W(indices)=1;
    end
    
    
end
end

