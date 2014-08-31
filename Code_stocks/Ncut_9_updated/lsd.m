function C = lsd(A, max_iter, W0,C_grnd_truth,K,print_every)
[n,r] = size(W0);

W0 = bsxfun(@rdivide, W0, sum(W0,2)+eps);

if n<8000
    c = abs(sum(sum(pinv(full(A)))))/r;
else
    [E,D,~] = svds(A,30);
    M = bsxfun(@rdivide, E, sqrt(diag(D))'+eps);
    c = norm(sum(M))^2/r;
end

W = W0;
for iter=1:max_iter
    gradn = c*A*W;
    gradp = W*(W'*W);
    a = sum(W./(gradp+eps),2);
    b = sum(W.*gradn./(gradp+eps),2);
    W = W .* (bsxfun(@times, gradn, a) + 1) ./ (bsxfun(@plus, bsxfun(@times, gradp, a), b)+eps);   
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if( mod(iter,print_every)==0 || iter==max_iter)
        
        [~,idx] = max(W,[],2);
        prt = purity(idx, C_grnd_truth, r);


        display(' ')
        display([' ITERATION ' , num2str(iter)])
        display([' purity of partition =  ' , num2str(prt)])
        display([' energy of partition =  ',     num2str(   compute_energy_of_partition(K,r,idx)  )  ])
        display( 'class sizes = ' )
           classSize(idx,r)       
        display(' ')
               
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
end

C=idx;


