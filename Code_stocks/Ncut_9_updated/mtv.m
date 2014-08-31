function [F,Classes,Energy,Energy_tresh,Time,iter] = mtv( K , L , F_init , rel_tol, maxTime,max_iter,C,verbose, print_every) 

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
    V = H_plus -   bsxfun(@times,H_minus,lambda) +   bsxfun(@times,H_zero,q);   %V=( v_1 , ... , v_R ) where v_i belongs to the subdifferential of B at f_i
    
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
       
while ( innerFlag == 0 )
    iter = iter + 1;   
    
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
    quantity = change_in_energy  - 0.99 * dist_btw_iterates;                  
    if( quantity>0 )
        innerFlag = 1;
    end;
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if( mod(iter,print_every)==0 && verbose==1 )
        
        [~,idx] = max(F,[],2);
        prt = purity(idx, C, R);
        
    display(' ')
    display([' ITERATION ' , num2str(iter)])
    display([ ' purity =                                ' , num2str(prt)])
    %display([ 'rel change in energy = ',  num2str(   abs(Total_Energy-Total_Energy_old)/Total_Energy   )  ])
    display([ ' energy of relaxed indicator functions = ',  num2str(   Total_Energy  )  ])  
    display([ ' energy of partition =                   ',     num2str(   compute_energy_of_partition(K,R,idx)  )  ])
    display( ' ' )
    display( ' class sizes = ' )
    classSize(idx,R)
        display(' ')
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end;
    
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
F_tresholded=sparse(1:n,Classes,1,n,R);
[~,~,E] = compute_HBE(K, lambda, F_tresholded); Energy_tresh = sum(E);
Energy=Total_Energy;

end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function  [H,B,E] = compute_HBE(K, lambda, F) 
    
    %Compute asymetric median of each relaxed indicator functions in F     
    n = size(F,1);
    k=ceil(  n/(1+lambda) );
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
    
