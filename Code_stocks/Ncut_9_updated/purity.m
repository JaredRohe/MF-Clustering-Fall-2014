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

