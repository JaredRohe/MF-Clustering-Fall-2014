clear all;
addpath('../Data_stocks');
addpath('Ncut_9_updated');
%% User Input
dataset=input('\nChose a Data Set:\n 1.) SP500\n 2.) Russell300\n');
if dataset==1
    sdataset=' SP500';
else
    sdataset=' Russell3000';
end
neighbors=input('\nGraph:\nEnter the number of nearest neighbors(0 for the full graph): ');
if neighbors~=0
    sneighbors=strcat(int2str(neighbors),'neighbors');
    
else
    sneighbors='-Full Graph';
    weight=0;
end
if neighbors~=0
    weight=input('\nWeight:\n 0.) Unweighted\n  1.) Weighted: ');
    if weight==1
        sweight='Weighted';
    else
        sweight='Unweighted';
    end
    
end

metric=input('\n 1.) CDF\n 2.) log returns\n ');
if metric==2
    metric='returns';
else
    metric='CDF';
end

window=input('\nEnter 0 for the full time period,\nor enter the size of the sliding window: '); 
if window==0
    window=1510;
end
alg=input('\nPlease Choose a clustering algorithm:\n 1.) Spectral Clustering \n 2.) MTV\n 3.) Kmeans\n');


fprintf('\nYou are about to cluster with the following specifications:\n');
disp(strcat('-Spectral Clustering on the ',sdataset));
disp(strcat('-',metric));
disp(sneighbors);
if neighbors~=0
    disp(sweight);
end
if window==1510
    disp('-Full Time Window')
else
    disp(strcat('Sliding Day Window: ', window, ' days'))

end
check=input('\nIs this correct?\n 1.) Yes\n 2.) No\n');
if check~=1
     return
end


if dataset==2
    %% Load Russell
    fprintf('\nClustering the Russell3000....')
    
    load('Russellraworg.mat');
    raworg2=theRussellraworg;
    
    load('Russell3000_Lookup.mat');
    LookUp=Russell3000_LookUp;
    
    load('Russell3000_IndustryMatrix.mat')
     Industries=Russell3000industries;
    % 
    % 
    % % % select subset of Russell to cluster:   %%%%%
    % % tempraworg=zeros(1510,64);
    % % newrusindustries=zeros(8,64);%% records industry of stock by its index w respect to new raworg matrix
    % % a=0;
    % % for i=1:8
    % %     industryindices=find(Russell3000industries(i,:));
    % % 
    % %     randomstocks= datasample(industryindices,8,'Replace',false);
    % %     
    % %      
    % %     tempraworg(:,a*8+1:8*i)=raworg2(:,randomstocks);
    % %     
    % %     
    % %     a=a+1;
    % % end
    % % 
    % % 
    % % i=1;
    % % for j=1:64
    % %     newrusindustries(i,j)=i;
    % %     if mod(j,8)==0
    % %          i=i+1;
    % %     end
    % % end
    % % 
    % %   
    % % % % fix clustermat_tickers at bottom Ncut discrete needs to be reindexed
    % % % with respect to raworg2:  **Create new RussellLookup**;
    % % 
    % % 
    % % raworg2=tempraworg;
    % % Russell3000industries=newrusindustries;
    % % %%%%%%%%%%

else
 %% Load S&P
    fprintf('\nClustering the SP500....');
    load('SP500raworg.mat');%use stockdataset.m to generate these mat files

    load('SP500_Lookup.mat');
    LookUp=SP500_LookUp;

    load('SP500_IndustryMatrix.mat');
    Industries=SP500industries;


end

%% Writing Files

%load('CRSPSP500dates07_12.mat'); %to label each file name


% mypath='/home/jlrohe/Desktop/Stock research project/StockClustering';
% mypath2='/home/jlrohe/Desktop/Stock research project/StockClustering/90daySP500modularity_CDF';
% 
% %formatSpecclusterid=('%dclusters:%d_idinfoSP500mcap.csv');
% formatSpecclusterid=('90dayclusteridSP500mcap_CDF.csv');
% %formatSpecclustereig=('%dclusters:%d_eigSP500mcap.csv');
% formatSpecmod=('90daysmodularitySP500mcap_CDF.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% General Info
[M,N]=size(raworg2);
stockcount=N;
days=M;

lastcluster=6;

[W,Rank,Dist,logreturns] = computeSimilarity(raworg2,neighbors,weight,metric);






%% Clustering
slidingwindow=zeros(window,stockcount);

tempmat=zeros(stockcount,lastcluster);
clustermat=zeros(stockcount,(lastcluster-1)*(days-window+1));
clustermat2=zeros(stockcount,(lastcluster)*(days-window+1));


modvec=zeros(1,lastcluster);
modmat=zeros(days-window,lastcluster);

purityvec=zeros(1,lastcluster);
puritymat=zeros(days-window,lastcluster);






windowcount=0;

for i=1:days-window+1
    for j=2:lastcluster
        slidingwindow(:)=raworg2(i:i+(window-1),:);
        
        [W,Rank,Dist,logreturns] = computeSimilarity(slidingwindow,neighbors,weight,metric);
        
        
        
%         [IDX,C,sumd,D] = kmeans(Rank',j);
%         tempmat(:,j)=IDX;
%         
        
        [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,j);
        [ clusterid ] = clusterID(NcutDiscrete,stockcount);
        Q=modularity_metric(W,clusterid,j);
        modvec(j)=Q;
        tempmat(:,j)=clusterid;
        
% 
%         [MTVsol,purity,energy,C]=MTV_clustering(W,clusterid,j);
%         purityvec(1,j)=purity;




                cluster_color = ['rgbmyc'];
                figure(1);clf;
                for k=1:j
                    id = find(NcutDiscrete(:,k));
                    plot(NcutEigenvectors(id,1),NcutEigenvectors(id,2),[cluster_color(k),'s'], 'MarkerFaceColor',cluster_color(k),'MarkerSize',5); hold on;
                    
                   
                end
                %hold off; axis image;
                %figure(2);
                %plot(NcutEigenvectors);



        %filename1=sprintf(formatSpecclusterid,dates(i),j);
        %csvwrite(fullfile(mypath,filename1),clusterid);

%         filename2=sprintf(formatSpecclustereig,dates(i),j);
%         csvwrite(fullfile(mypath,filename2),NcutEigenvectors);
%         dlmwrite(fullfile(mypath,filename2),NcutEigenvalues,'-append');

    end
    tempmat2=tempmat;
    tempmat(:,1)=[]; 

    clustermat(1:stockcount,1+((lastcluster-1)*windowcount):(lastcluster-1)*(windowcount+1))=tempmat;
    clustermat2(1:stockcount,1+(lastcluster*windowcount):lastcluster*(windowcount+1))=tempmat2;
    
    windowcount=windowcount+1;


    modmat(i,:)=modvec; 
    puritymat(i,:)=purityvec;

end


% filename1=sprintf(formatSpecclusterid);
% csvwrite(fullfile(mypath,filename1),clustermat);
%  
% filename3=sprintf(formatSpecmod);
% csvwrite(fullfile(mypath2,filename3),modmat);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

%% Breakdown
[~,C]=size(clustermat);


[numindustries,~]=size(Industries);
inddistribution=zeros(numindustries*C,lastcluster);%make sure row number is the correct number of industries!



group=zeros(numindustries,lastcluster);% will contain the number of selected industry stocks per each cluster % zero will be first element (remove after)



indhelp=0;
for i=1:C %each col in clustermat
    
    for j=1:numindustries %indices of each industry/row in industrymatrix
        for k=1:stockcount% add corresponding indices to the group variable which is a counter
            if Industries(j,k)~=0 %located a stocked in the jth industry
                groupind=clustermat(k,i);%which cluster is the stock in
                group(j,groupind)=group(j,groupind)+1;%for 1 col in cluster mat counts percent of industries present in each cluster
            else
                continue
            end
        end

         group(j,:)=group(j,:)./sum(group(j,:));% percentage of industry

    end
    inddistribution(indhelp*numindustries+1:numindustries*i,1:lastcluster)=group;
    indhelp=indhelp+1;
    group(:)=0;
    
    
end



%%% tickers of stocks in each cluster****

clustermat_tickers=cell(stockcount,lastcluster);
for i=1:lastcluster
    stockindices=find(NcutDiscrete(:,i));
    tickers=LookUp(stockindices,2);
    [A,~]=size(tickers);
    clustermat_tickers(1:A,i)=tickers;
    

end
    

%% Vizualization

% Z = linkage(Rank,'ward','euclidean','savememory','on');
% tree=Z;
% p=1500;
% dendrogram(tree,p)

% [coeff,score,latent,tsquared,explained,mu]=pca(Rank,'NumComponents',2);

% %%%% Some visualization code that was in stock Clustering program.m after ncut function call
 
% clustering graph in



% display clustering result


