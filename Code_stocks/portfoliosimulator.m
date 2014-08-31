%% User Input
disp('Portfolio Simulator');
dataset=input('Chose a Data Set-\n For SP500: 1 \n For Russell3000: 2\n');
neighbors=input('Enter 0 for full graph, \n or Enter the number of nearest neighbors: ');
if neighbors==0
    disp('Full Graph')
else
    disp(strcat(int2str(neighbors), ' nearest neighbors'));
end

load('CRSPSP500dates07_12.mat');

if dataset==2
    %% Load Russell
    disp('Clustering the Russell3000....')
    
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
    disp('Clustering the SP500....')
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



%% REQUIRED INPUT
[M,N]=size(raworg2);
stockcount=N;
days=M;

window=500;   % Full Window: 1510
lastcluster=4;
weight=0; %1 = weighted, 0 =unweighted
metric='CDF'; %'returns' or 'CDF'



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cash=1000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Clustering
slidingwindow=zeros(window,stockcount);


clustermat=zeros(stockcount,(lastcluster-1)*(days-window+1));
clustermat2=zeros(stockcount,(lastcluster)*(days-window+1));

%%%%%%%
modvec=zeros(1,lastcluster);
%%%%%%%


modmat=zeros(days-window,lastcluster);

purityvec=zeros(1,lastcluster);
puritymat=zeros(days-window,lastcluster);






windowcount=1;
startdate=125;  %07/02/07    %change start date for diff. size window
while startdate+window<=1510
 
    
        slidingwindow(:)=raworg2(startdate:startdate+(window-1),:);
        
        [W,Rank,Dist] = computeSimilarity(slidingwindow,neighbors,weight,metric);
        
        for k =2:lastcluster
        
    %         [IDX,C,sumd,D] = kmeans(Rank',j);
    %         tempmat(:,j)=IDX;
    %         

            [NcutDiscrete,~,~] = ncutW(W,k);
            [ clusterid ] = clusterID(NcutDiscrete,stockcount);
            Q=modularity_metric(W,clusterid,k);
            modvec(k)=Q;
%             clustermat(:,windowcount)=clusterid;


    %         [MTVsol,purity,energy,C]=MTV_clustering(W,clusterid,numcluster);
    %         purityvec(1,numcluster)=purity;


        end
        
        [~,numcluster]=max(modvec); % number of clusters according to modularity
        [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,numcluster);
        [ clusterid ] = clusterID(NcutDiscrete,stockcount);
        
        
        %%% allocation %%%
        cashpercluster=cash/numcluster;
        
        numstockspercluster=sum(NcutDiscrete);
        allocation=cashpercluster./numstockspercluster;
%        returnfactor=
        

       

    %clustermat(1:stockcount,1+((numcluster-1)*windowcount):(numcluster-1)*(windowcount+1))=tempmat;
    %clustermat2(1:stockcount,1+(numcluster*windowcount):numcluster*(windowcount+1))=tempmat2;
    
   


%     modmat(i,:)=modvec; 
%     puritymat(i,:)=purityvec;
%     
    
    
    
    
    
   
    
    
    
    
    
%     
%     startdate=
      windowcount=windowcount+1;
    
    

end

return
% filename1=sprintf(formatSpecclusterid);
% csvwrite(fullfile(mypath,filename1),clustermat);
%  
% filename3=sprintf(formatSpecmod);
% csvwrite(fullfile(mypath2,filename3),modmat);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

% %% Breakdown
% [~,C]=size(clustermat);
% 
% 
% [numindustries,~]=size(Industries);
% inddistribution=zeros(numindustries*C,lastcluster);%make sure row number is the correct number of industries!
% 
% 
% 
% group=zeros(numindustries,lastcluster);% will contain the number of selected industry stocks per each cluster % zero will be first element (remove after)
% 
% 
% 
% indhelp=0;
% for i=1:C %each col in clustermat
%     
%     for j=1:numindustries %indices of each industry/row in industrymatrix
%         for k=1:stockcount% add corresponding indices to the group variable which is a counter
%             if Industries(j,k)~=0 %located a stocked in the jth industry
%                 groupind=clustermat(k,i);%which cluster is the stock in
%                 group(j,groupind)=group(j,groupind)+1;%for 1 col in cluster mat counts percent of industries present in each cluster
%             else
%                 continue
%             end
%         end
% 
%          group(j,:)=group(j,:)./sum(group(j,:));% percentage of industry
% 
%     end
%     inddistribution(indhelp*numindustries+1:numindustries*i,1:lastcluster)=group;
%     indhelp=indhelp+1;
%     group(:)=0;
%     
%     
% end
% 
% 
% 
% %%% tickers of stocks in each cluster****
% 
% clustermat_tickers=cell(stockcount,lastcluster);
% for i=1:lastcluster
%     stockindices=find(NcutDiscrete(:,i));
%     tickers=LookUp(stockindices,2);
%     [A,~]=size(tickers);
%     clustermat_tickers(1:A,i)=tickers;
%     
% 
% end
    

%% Vizualization

% Z = linkage(Rank,'ward','euclidean','savememory','on');
% tree=Z;
% p=1500;
% dendrogram(tree,p)

% [coeff,score,latent,tsquared,explained,mu]=pca(Rank,'NumComponents',2);

% %%%% Some visualization code that was in stock Clustering program.m after ncut function call
 
% % clustering graph in
% % figure(3);
% % plot(NcutEigenvectors);
% 
% 
% % % display clustering result
% % cluster_color = ['rgbmyc'];
% % figure(2);clf;
% % for j=1:nbCluster,
% %     id = find(NcutDiscrete(:,j));
% %     plot3(data(1,id),data(2,id),data(3,id),[cluster_color(j),'s'], 'MarkerFaceColor',cluster_color(j),'MarkerSize',5); hold on; 
% % end
% % hold off; axis image;
% % disp('This is the clustering result');
% % disp('The demo is finished.');
% 
