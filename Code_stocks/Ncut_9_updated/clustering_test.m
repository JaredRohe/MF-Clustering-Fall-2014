
function clustering_test(A,Cgrnd_truth,nc)

%%%%%%%%%%%%%%%%%%
max_iter = 20000; % total number of iterations
print_every=500; % print purity and energy every "print_every" iteration
%%%%%%%%%%%%%%%%%


A=double(A);
W=max(A,A');
%W=double(W>0);
R=nc;
n = size(W,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters 
% default: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nb_of_anchors_per_class=1;  % default value = 1
             


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the gradient matrix K        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

triangularW=triu(W,1);
[I, J, v]=find(triangularW);
m=length(v); %number of edges in the graph
KI=[1:m 1:m];
KJ(1:m) = I;
KJ(m+1:2*m) = J;
Kv(1:m)=v;
Kv(m+1:2*m)=-v;
K=sparse(KI,KJ,Kv,m,n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the Laplacian matrix Lap     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = sum(W,2);
Lap = spdiags(D,0,n,n) - W;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do Shi-Malik Normalized Cut Algorithm  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath Ncut_9; nSeed=0;
[NcutDiscrete,~,~] =ncutW(W,R,nSeed);
[~,C_ncut] = max(NcutDiscrete,[],2);
save('C_ncut.mat','C_ncut');

    prt = purity(C_ncut, Cgrnd_truth, nc);
    enrg = compute_energy_of_partition(K,R,C_ncut);
    display( ' ' )
    display([' purity of Ncut partition = ' , num2str(prt)])
    display([' energy of Ncut partition = ' , num2str(enrg)])
    display( ' ' )
    display( 'class sizes of Ncut = ' )
    classSize(C_ncut,nc)
    display(' press a key to start NMFR clustering ')
    pause
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


F_init = zeros(n,R);


% Pick indices of anchors at random from the previous partition
[indices_of_anchors, indicator_matrix_of_anchors  ] = pick_random_indices_from_classes(C_ncut, R, nb_of_anchors_per_class);
%[indices_of_anchors, indicator_matrix_of_anchors  ] = pick_random_indices_from_classes(Cgrnd_truth, R, nb_of_anchors_per_class);

% Set the indicator function of cluster r to be one on the anchors
F_init(indices_of_anchors,:) = indicator_matrix_of_anchors;

% Ciffuse these indicator functions
F_init=diffuse(Lap, F_init);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NMFR CLUSTERING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
 check_step = 1000;
 nSeed = 0;
 verbose = false;
%verbose = true;
 candidate_alphas = [linspace(0.1,0.9,9),0.99];   
 An = Normalize_Similarity_2(W);
 
                       
 C_nmfr = nmfr_auto(An, nc, candidate_alphas, max_iter, check_step, verbose, F_init+0.2, nSeed, Cgrnd_truth,K,print_every);
 save('C_nmfr.mat','C_nmfr');




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



