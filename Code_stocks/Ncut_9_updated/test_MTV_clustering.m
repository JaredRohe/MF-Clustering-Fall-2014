
% ----------------------------------------------
% Multiclass Total Variation Clustering (MTV Clustering)
% X. Bresson, T. Laurent, D. Uminsky and J.H. von Brecht, 
% Annual Conference on Neural Information Processing Systems (NIPS), 2013 
% ----------------------------------------------



function test_MTV_clustering


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMMENT OR UNCOMMENT DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 4 MOONS (4,000 data points)
load('data/FOURMOONS.mat','A','C','nc');
[MTVsol,purity,energy] = MTV_clustering(A,C,nc,'FOUR_MOONS');


% WEBKB4 (4,196 data points)
load('data/WEBKB4.mat','A','C','nc');
[MTVsol,purity,energy] = MTV_clustering(A,C,nc,'WEBKB4');


% OPTDIGITS (5,620 data points)
load('data/OPTDIGITS.mat','A','C','nc');
[MTVsol,purity,energy] = MTV_clustering(A,C,nc,'OPTDIGITS');


% PENDIGITS (10,992 data points)
load('data/PENDIGITS.mat','A','C','nc');
[MTVsol,purity,energy] = MTV_clustering(A,C,nc,'PENDIGITS');


% 20NEWS (19,938 data points)
load('data/20NEWS.mat','A','C','nc');
[MTVsol,purity,energy] = MTV_clustering(A,C,nc,'20NEWS');


% MNIST (70,000 data points)
load('data/MNIST.mat','A','C','nc');
[MTVsol,purity,energy] = MTV_clustering(A,C,nc,'MNIST');



end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MTVsol,purity,energy] = MTV_clustering(A,Cgrnd_truth,nc,name_dataset)

fprintf('\nDataset= %s, Nb data points= %i, Nb classes= %i\n\n',name_dataset,size(A,1),nc);


W=max(A,A');
W=double(W>0);
R=nc;
n = size(W,2);

verbose = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters 
% default: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nb_of_anchors_per_class = 1;  % default value = 1
maxIter = 2000;               % default value = 2000
maxTime = 1e10;               % default value = infinty 
rel_tol = 1e-4;               % default value = 0
nbTrials = 30;                % number of random trials (30-100)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the gradient matrix K        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('Constructing the Gradient matrix and estimating its norm')
triangularW=triu(W,1);
[I, J, v]=find(triangularW);
m=length(v); %number of edges in the graph
KI=[1:m 1:m];
KJ(1:m) = I;
KJ(m+1:2*m) = J;
Kv(1:m)=v;
Kv(m+1:2*m)=-v;
K=sparse(KI,KJ,Kv,m,n);
L=normest(K);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the Laplacian matrix Lap     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('Computing Laplacian matrix')
D = sum(W,2);
Lap = spdiags(D,0,n,n) - W;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do Shi-Malik Normalized Cut Algorithm  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('Computing Ncut solution');
addpath Ncut_9; nSeed=0;
[NcutDiscrete,~,~] =ncutW(W,R,nSeed);
[~,C_ncut] = max(NcutDiscrete,[],2);

 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO TV CLUSTERING MULTIPLE TIMES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('Computing MTV solution');

C_best = [];
Energy_best = 1e10;
purity_best = 1e10;


% Random initialization
display(' ');
display('Trials');
for k=1:nbTrials
    
    F_init = zeros(n,R);
    
    % Pick indices of anchors at random from the previous partition
    [indices_of_anchors, indicator_matrix_of_anchors  ] = pick_random_indices_from_classes(C_ncut, R, nb_of_anchors_per_class);
    
    % Set the indicator function of cluster r to be one on the anchors
    F_init(indices_of_anchors,:) = indicator_matrix_of_anchors;
    
    % Ciffuse these indicator functions
    F_init=diffuse(Lap, F_init);
    
    % Run the TV-clustering algorithm with these initial data
    [~,C,~,Energy_tresh,~,~,purity] =  TVclust(K, L, F_init, rel_tol, maxTime, maxIter, Cgrnd_truth, verbose);
    
    % test purity
    if Energy_tresh< Energy_best
        C_best = C;
        Energy_best = Energy_tresh;
        purity_best = purity;
    end
    
    kTrial = k
    Energy = Energy_tresh
    Energy_best
    purity
    purity_best
    pause(1)
    
end




MTVsol = C_best;
purity = purity_best;
energy = Energy_best;


fprintf('Purity= %.4f for dataset= %s\n',purity,name_dataset);

display(' ');
display('Press any key to continue to the next dataset');
display(' ');
pause;


end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_F = diffuse(Lap, F)
% Diffuse each column of F data by doing one step of implicit heat equation:
%                     (I+dt Lap) f_r^{new} = f_r    r=1,...,R

dt=1;
[n,R]=size(F); new_F=zeros(n,R);
md=@(x,type) x+dt*Lap*x; 
for i=1:R,
    f1 = F(:,i);
    [f1, ~] = pcg(md,f1,1e-6,50,[],[],f1);
    new_F(:,i) = f1;
end;

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [idx, indicator_matrix]= pick_random_indices_from_classes(C,R,nb_per_class)
% idx is a column vectors of heights nb_per_class*R. 
% The first nb_per_class entries contains indices randomly picked from class 1, 
% The next nb_per_class  entries contains indices randomly picked from class 2, etc...
rng('default'); rng('shuffle');
idx=zeros(nb_per_class*R,1);
indicator_matrix=zeros(nb_per_class*R,R);

for k=1:R 
  idx_in_class_k= find(C==k);
  ClassSize=length(idx_in_class_k);
  if (ClassSize==0)
      display('one of the class is empty!')
  end
  v = randi(  [1 , ClassSize]  ,  nb_per_class , 1 );
  selected_idx = idx_in_class_k(v);
  idx(  (k-1)*nb_per_class +1 :  k*nb_per_class  ) = selected_idx;
  indicator_matrix((k-1)*nb_per_class +1 :  k*nb_per_class , k)=1;
end

end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [F,C,Energy,Energy_tresh,Time,iter,prt] = TVclust( K , L , F_init , rel_tol, maxTime, max_iter, C, verbose) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Find a local min of
%         TV(f_1) / B(f_1) +...+ TV(f_R) / B(f_R) 
%         where B(f)=|| f - med_lambda(f) ||_1
%   subject to the simplex constraint f_1 + ... + f_R = 1, f_1, ..., f_R >=0
%        
%   INPUT :
%       K           - the  gradient matrix of a graph (an NxN matrix)
%       L           - norm of the gradient matrix: L=normest(K)
%       F_init      - the inital guess for the R relaxed indicator functions (an NxR matrix) 
%       rel_tol     - relative tolerance for the energetic stopping criteria
%
%   OUTPUT :       
%       F      - the R relaxed indicator functions (an NxR matrix)
%       C      - the partition associated to F  (an Nx1 matrix)
%       Energy - the discrete energy of the computed partition
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
Tstart=tic; 
[n,R] = size(F_init);              % number of data points and number of classes 
m=size(K,1);                       % number of edges in the graph 
lambda = 1/(R-1);                  % parameter for the asymetric balance term
F = Project_to_Simplex(F_init);    % project F_init on the simplex  
P=zeros(m,R);                      % initial choice for the dual variable
tau=1/L;                           % initial choice for the time step in the inner loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                BEGINNING OF OUTER LOOP              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
%compute balance term, total variation and energy of each classes of the initial iterate                              
[H,B,E] = compute_HBE(K, lambda, F); 


Total_Energy=1000;
iter = 0;flag = 0;
while( flag == 0)
    
    Delta = max(B);    
    Total_Energy_old= Total_Energy;
    Total_Energy = sum(E);   % Sum of the energy of each class
        
    % Compute the subdifferential of the balance term 
    H_plus = (H>0); n_plus = sum(H_plus,1);
    H_minus = (H<0); n_minus = sum(H_minus,1);
    H_zero = (H==0); n_zero = sum(H_zero,1);
    q = (lambda.*n_minus - n_plus)./n_zero;     
    V = H_plus - bsxfun(@times,H_minus,lambda) + bsxfun(@times,H_zero,q);   %V=( v_1 , ... , v_R ) where v_i belongs to the subdifferential of B at f_i
    
    % forward step
    G = F + bsxfun(@times, Delta*V, E./B); 
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%        INNER LOOP              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    B_outer=B; F_outer=F; E_outer=E;   
    Gamma = Delta./B;
    L_tilda= max(Gamma)*L;  sigma=1/(tau*L_tilda^2);   
    F_bar = F;
    innerFlag = 0; 
    innerIter = 0; %NEW
    
while ( innerFlag == 0 )
    iter = iter + 1;
    innerIter = innerIter + 1; %NEW
    
    % Chambolle-Pock Primal Dual algorithm       
    P = P + sigma * bsxfun(@times, K*F_bar, Gamma);  
    P = P./max(1,abs(P));    
    F_old = F;
    F = F - tau * bsxfun(@times, K'*P, Gamma);    
    F = (F + tau*G)/(1+tau);
    F = Project_to_Simplex(F);
    theta=1/sqrt(1+2*tau); tau=theta*tau;  sigma=sigma/theta;   
    F_bar= (1+theta) * F - theta * F_old;   
                    
    % Compute H, B, T, and E  associated to the current iterate of the PD algorithm
    [H,B,E] = compute_HBE(K, lambda, F); 
                                    
    % Stopping criteria for the inner loop
    change_in_energy = sum( Delta * (B./B_outer) .* (E_outer - E)  );                             
    dist_btw_iterates = norm(  F - F_outer, 'fro' )^2;
    quantity = change_in_energy  - 0.999 * dist_btw_iterates;  % 0.99             
    if( quantity>0 || innerIter>100 ) %NEW
        innerFlag = 1;
    end;
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %if( mod(iter,250)==0 && verbose==1 )
    if( mod(iter,max_iter)==0 && verbose==1 )
        
        [~,idx] = max(F,[],2);
        prt = purity(idx, C, R);
        
        display(' ')
        display(['ITERATION ' , num2str(iter)])
        display(['Purity = ' , num2str(prt)])
        display([ 'Relative change in energy = ',  num2str(   abs(Total_Energy-Total_Energy_old)/Total_Energy   )  ]);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%      END OF INNER LOOP  // STOPPING CREITERIA FOR OUTER LOOP         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
     Time=toc(Tstart);
    
    if( abs(Total_Energy-Total_Energy_old)/Total_Energy <= rel_tol || Time>= maxTime || iter>=max_iter )
        flag = 1; 
    end
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%   END OF OUTER LOOP  //  COMPUTE OUTPUT VARIABLES         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,Classes] = max(F,[],2);
F_tresholded = sparse(1:n,Classes,1,n,R);
[~,~,E] = compute_HBE(K, lambda, F_tresholded); Energy_tresh = sum(E);
Energy = Total_Energy;
prt = purity(Classes, C, R);

toc
display(' ');

end
 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function  [H,B,E] = compute_HBE(K, lambda, F) 
    
    %Compute asymetric median of each relaxed indicator functions in F     
    n = size(F,1);
    k = ceil(  n/(1+lambda) );
    Z = sort(F,1);
    M = Z(k,:);
   
    % compute H, B ,T and E
    H=bsxfun(@minus,F,M);
    tilted_abs_H = H.*(H>0) - lambda*H.*(H<0);
    B=sum( tilted_abs_H , 1); % row vector containing the balance term of each class    
    T = sum ( abs(K*F) , 1 ); % row vector containing the balance term of each class     
    E = T./B;                 % row vector containing the Energy for each class
    end

    
    
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function x = Project_to_Simplex(F)
% F is a matrix with N rows of length R
% Each of the N rows is projected onto the canonical simplex in R^R.  

[N,R] = size(F);
x = F;
n_I = R*ones(N,1);
I = zeros(N,R);
flag = zeros(N,R);

while(min(min(flag)) == 0)    
    y = ( sum(x,2)-1 )./n_I;
    x=bsxfun(@minus, x,y).*(1-I);
    
    flag = (x>=0);
    I = I + (x<0);
    n_I = R - sum(I,2);
    x = x.*(x>=0);   
end
    
end   



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function [ percentage] = purity(computed_labels, actual_labels , R)

N=length(actual_labels);

m=0;

for k=1:R
   index=find(computed_labels==k);
   
   
   % v is a vector describing the actual labels of the data points in cluster k.   
   v= actual_labels(index);  
  
   
   % num(l) is the number of data samples in the cluster k that belong to ground-truth class l
   num=zeros(1, R);
   for l=1:R
       num(l)= length(  find(v==l)  );
   end
    
   m=m+max(num);
    
end

percentage=(m/N)*100;


end



    



