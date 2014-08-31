%% Read in raw data

% replace importdata calls with appropriate data sets and the library
% save desired variables, then load to window clustering and run script
% from window clustering. 


% in case importdata command doesnt import file as a structure
uiimport('rawSP500ref.csv'); 
 ref=[rawSP500ref(:,1),rawSP500ref(:,3)];


ref1=importdata('rawSP500marketcapref.csv'); %file containing tickers

ref=[ref1.textdata(2:end,1),ref1.textdata(2:end,3)];% assumes permnos in first column and ticker in 3rd column
raw=csvread('rawSP500marketcap.csv',1,0);% main data set

raw=abs(raw);


% Russell data%%%%


% load('THErussellraworg_mcap.mat');
% load('theRussellLookup.mat');
% load('theRussellLibary.mat');
% 
% 
% [M,N]=size(theRussellraworg);
% 
% 
% stockcount=N;
% 










%% Data set check

% K=logical(raw);
% sparse=numel(K)-sum(K(:));
% 
% if sparse~=0
%     warning('Some stocks have missing data.')
% end

%% Library lookup

% indlibrary1=importdata('tickindustrylibrary.xlsx'); %tickindustrylibrary is library for SP500
% indlibrary=strtrim(indlibrary1);
% 
% library=cell(461,2);
% 
% 
% for i=1:461
%     library{i,1}=indlibrary(1,(2*i+1)-2);
% end
% 
% for j=1:461
%     library{j,2}=indlibrary(1,2*j);
% end


  
%% Raw Data Info
% 
% stocklables=unique(raw(:,1));
% stocklables=stocklables';
% 
% 
% markcap=raw(:,3).*raw(:,4);
% raw=[raw,markcap];
% 
% [m,n]=size(raw);
% 
% % determines number of stocks in dataset
% stockcount=1;
% stock=raw(1,1);
% for i=2:m
%     if stock~=raw(i,1);
%         stockcount=stockcount+1;
%         stock=raw(i,1);
%     end
% end
% 
% AAPL=14593; % check this number if data set is using something other than PERMNO
% days=0;
% for i=1:m
%     if raw(i,1)==AAPL;
%         days=days+1;
%     end
% end

%% Organize dates as first column and subsequent columns as each stock

% raworg=zeros(days,stockcount+1);
% 
% raworg(:,1)=raw(1:days,2);% first column is dates
% 
% k=1;
% p=1;
% for i=1:stockcount
%     raworg(:,i+1)=raw(k:days*p,5);%pulls off just markcap from raw
%     k=k+days;
%     p=p+1;
% end
% 
% dates=raworg(:,1);
% 
% raworg=raworg(:,2:end);
% 
% [M,N]=size(raworg);

 %% Lookup
% 
% %orders stocks on when they appear in the raw data set
% Lookup=cell(stockcount,1);
% permno=ref{1,1};
% count=1;
% for i = 2:m
%     if isequaln(ref{i,1},permno)==0
%         Lookup{count,1}=ref{i-1,2};
%         count=count+1;
%         permno=ref{i,1};
%     end
% end
% 
% Lookup{count,1}=ref{m,2};


%% Ordered Library

% orderedlibrary=cell(stockcount,1);
% for i =1:stockcount
%     for j=1:stockcount
%         if ismember(theRussellLookup{i,1},LIBRARY{j,1})==1
%             orderedlibrary{i,1}=LIBRARY{j,2};
%         end
%             
%         
%     end
% end
% 
% industry='Consumer Goods';
% Russellcongoods=zeros(stockcount,1);
% for i =1:stockcount
%     if ismember(orderedlibrary{i},industry)==1
%         Russellcongoods(i,1)=i;
%     end
%     
% end



        
       

%% Change time window

% raworg=[dates,raworg];
% 
% start=20070103;
% finish=20070511;
% 
% sindex=0;
% findex=0;
% for i=1:M
%     if raworg(i,1)==start
%         sindex=i;
%     end
% end
% for j=1:M        
%     if raworg(j,1)==finish
%         findex=j;
%     end
% end
% raworg2=raworg(:,2:end);
% newraworg=raworg2(sindex:findex,:);

