function  total_energy = compute_energy_of_partition(K,R, C) 
% C must be a column vectors!

n = size(C,1);
F=zeros(n,R);

if(n ~= size(K,2)  )
display('error in compute_energy_of_partition: C is not a column vector!');
end

for r=1:R
F(:,r) = (C==r);
end
   
    lambda = 1/(R-1);
    
    %Compute asymetric median of each relaxed indicator functions in F     
   
    k=ceil(  n/(1+lambda) );
    Z = sort(F,1);
    M = Z(k,:);
   
    % compute H, B ,T and E
    H=bsxfun(@minus,F,M);
    tilted_abs_H = H.*(H>0) - lambda*H.*(H<0);
    B=sum( tilted_abs_H , 1); % row vector containing the balance term of each class    
    T = sum ( abs(K*F) , 1 ); % row vector containing the balance term of each class     
    E = T./B;                 % row vector containing the Energy for each class
    total_energy=sum(E);
    end
