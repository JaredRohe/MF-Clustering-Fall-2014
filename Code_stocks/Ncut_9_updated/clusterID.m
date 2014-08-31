function [ clusterid ] = clusterID(NcutDiscrete,stockcount)
%% ClusterID

[A,~]=size(NcutDiscrete);

%Which stocks are in which cluster
clusterid=zeros(stockcount,1);
for i=1:A
    clusteridindex=find(NcutDiscrete(i,:));
    clusterid(i,1)=clusteridindex;
end


end

