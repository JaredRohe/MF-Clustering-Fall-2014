load('SP500financeindexes.mat');
load('SP500techindices.mat');
load('SP500indgoodsindices.mat');
load('SP500congoodsindices.mat');
load('SP500servicesindices.mat');
load('SP500utilitiesindices.mat');
load('SP500healthcareindices.mat');
load('SP500BMindices.mat');


%% TICKER TRACKER %%%
% 
% [~,numfinance]=size(find(orderedliblogical));
% [~,numtech]=size(find(techorderedliblogical));
% [~,numindgoods]=size(find(indgoodsliblogical));
% [~,numcongoods]=size(find(congoodsliblogical));
% [~,numservices]=size(find(servicesliblogical));
% [~,numutililities]=size(find(utilitiesliblogical));
% [~,numhealthcare]=size(find(healthcareliblogical));
% [~,numBM]=size(find(BMliglogical));
% [~,numconglom]=size(find(conglomliglogical));
% 
% financeticktrack=repmat({''},numfinance,lastcluster);
% techticktrack=repmat({''},numtech,lastcluster);
% indgoodsticktrack=repmat({''},numingoods,lastcluster);
% congoodsticktrack=repmat({''},numcongoods,lastcluster);
% servicesticktrack=repmat({''},numservices,lastcluster);
% utilitiesticktrack=repmat({''},numutilities,lastcluster);
% healthcaretick=repmat({''},numhealthcare,lastcluster);
% BMticktrack=repmat({''},numBM,lastcluster);
% conglomticktrack=repmat({''},numconglom,lastcluster);
% 
% 
% allfinaceticktrac=repmat({''},numfinance*C,lastcluster);
% alltechticktrack=repmat({''},numtech*C,lastcluster);
% allindgoodsticktrack=repmat({''},numindgoods*C,lastcluster);
% allcongoodsticktrack=repmat({''},numcongoods*C,lastcluster);
% allservicesticktrack=repmat({''},numservices*C,lastcluster);
% allutilitiestrack=repmat({''}, numutilities*C,lastcluster);
% allhealthcaretick=repmat({''},numhealthcare,lastcluster);
% allBMticktrack=repmat({''},numBM,lastcluster);
% conglomticktrack=repmat({''},numconglom,lastcluster);
% 
% 
% indhelpf=1;
% for i=1:C %each col in clustermat
%     for j=1:numindustries %indices of each industry/row in industrymatrix
%         for k=1:stockcount% add corresponding indices to the group variable which is a counter
%             if(j==1)
%                 if industrymatrix(j,k)~=0
%                     currentfin=clustermat(k,i); %current fin is the cluster number for the current finance stock
%                     financeticktrack{indhelp,currentfin}=Lookup{k,1};
%                     inhelp=indhelp+1;
%                    
%                 else
%                     continue
%                 end
%             end
%             if(j==2)
%                  if industrymatrix(j,k)~=0
%                 techticktrack{;
%                 group(j,groupind)=group(j,groupind)+1;%for 1 col in cluster mat counts percent of industries present in each cluster
%                 else
%                     continue
%                 end
%             end
%             if(j==3)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;%for 1 col in cluster mat counts percent of industries present in each cluster
%                 else
%                     continue
%                 end
%             end
%             if(j==4)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;%for 1 col in cluster mat counts percent of industries present in each cluster
%                 else
%                     continue
%                 end
%             end
%             if(j==5)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;
%                 else
%                     continue
%                 end
%             end
%             if(j==6)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;
%                 else
%                     continue
%                 end
%             end
%             if(j==7)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;
%                 else
%                     continue
%                 end
%             end
%             if(j==8)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;
%                 else
%                     continue
%                 end
%             end
%             if(j==9)
%                  if industrymatrix(j,k)~=0
%                 groupind=clustermat(k,i);
%                 group(j,groupind)=group(j,groupind)+1;
%                  else
%                     continue
%                 end
%             end
%         end
%         
%         [~,tempsize]=size(find(industrymatrix(j,:)));
%         %group(j,:)=group(j,:)./tempsize;% percentage of industry
%     end
%     indhelp2=0;
%     allfinaceticktrac(1+numfinance*indhelp2:numfinance*i,lastcluster)=financeticktrack;
%     financeticktrack{:}='';
%     
%     alltechticktrack(1+numtech*indhelp2:numtech*i,lascluster)=tecktrack;
%     techticktrack{:}='';
%     
% end
% % 
% % % 
% % % finish this for rest of industries
% % % 
% % % 
% % % adjust the indhelp industries!!! they are reassigned each time through the loop.
% % % consider having each industry be stripped of zeros then put into a new matrix with zeros on end to make dimensionally compatible.
% % % Then if statement would be if(j==1and<max num of cols of this new industry max) then add the finance stock to fnace tick track
