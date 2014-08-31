function  v = classSize(C,nc)
v=zeros(1,nc);
for k=1:nc    
    v(k)=sum(C==k);
end

