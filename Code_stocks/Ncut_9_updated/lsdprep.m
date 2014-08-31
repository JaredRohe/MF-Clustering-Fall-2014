function C = lsdprep( slidingwindow,numcluster,alg,neighbors,groundtruth)
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


%% Computing Distance

Dist=zeros(stockcount,stockcount);

for i=1:N
    for j=i+1:N
        v1=Rank(:,i);
        v2=Rank(:,j);
        
        EDist= sqrt(sum((v2 - v1).^2));
        
        Dist(i,j)=EDist;
        Dist(j,i)=EDist;
        
    end
end

csvwrite('distmatrixSP500.csv',Dist);

%% Similarity Matrix

[W] = compute_relation4(Dist);

% Sparsifiy


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
end



%% Clustering algorithms

max_iter=10000;

C = lsd(W, max_iter, W0,groundtruth,K,print_every);


%% ClusterID
 
[A,~]=size(NcutDiscrete);

%Which stocks are in which cluster
clusterid=zeros(stockcount,1);
for i=1:A
    clusteridindex=find(NcutDiscrete(i,:));
    clusterid(i,1)=clusteridindex;
end


   
%% Modularity

lambda=1;

D=zeros(stockcount,1);
for i =1:stockcount
    D(i,1)=sum(W(i,:));
end

Q=0;
m=sum(D);
m=m/2;
for i=1:stockcount
    for j=i+1:stockcount-1
        findstock=find(NcutDiscrete(i,:));
        if find(NcutDiscrete(j,:))==findstock
            delta=1;
        else
            delta=0;
        end
        Q=Q+(1/(2*m))*(sum(W(i,j)-D(i,1)*D(j,1)/(2*m))*delta);
    end

end




%% F-Statistic

% Within Group
% GM=mean(Rank,2);
% withinvariation=0;
% betweenvariation=0;
% 
% 
% for i =1:numcluster
%     clusterindices=find((clusterid)==i);
%     [G,~]=size(clusterindices);
%     tempmatrix=zeros(M-1,G);
%     indhelp=1;
%     for j=clusterindices'
%         tempmatrix(:,indhelp)=Rank(:,j); %columns of Rank matrix that correspond to the stocks in each cluster
%         indhelp=indhelp+1;
%     end % all stocks of the jth cluster extracted from rank and stored in tempmatrix
%     
%     
%     withinmean=mean(tempmatrix,2);
%     betweenvec=withinmean-GM;
%     betweenvariation=betweenvariation+(norm(betweenvec).^2);
%  
%     for k=1:G
%         tempmatrix(:,k)=tempmatrix(:,k)-withinmean;
%         withinvariation=withinvariation+(norm(tempmatrix(:,k)).^2);
%        
%     end
%     
% end
% 
% betweenvariation=betweenvariation/(numcluster-1);
% withinvariation=withinvariation/(stockcount-numcluster);
% 
% F=betweenvariation/withinvariation;


