function [weights C] = sample_weights_add(C, weights, M, alpha, phi, tau, gvec)
[K T] = size(weights);


% For weights that stop appearing after t
for t=T-1:-1:1
    
    
    ind = find(sum(M(:, t+1:end), 2) ==0);
    wold = weights(ind, t+1:end);
    wmint = weights(ind, t);
    
    cold = C(ind, t:end-1);

    cprop = zeros(size(cold));
    
    cprop(:, 1)=  poissrnd(phi.*wmint);
    wprop = zeros(size(wold));
    it=1;
    
    
    for p = t:T-1
        wprop(:, it) = (cprop(:, it)>0).*gamrnd(cprop(:, it), 1/(phi+tau));
        cprop(:, it+1)=  poissrnd(phi.*wprop(:, it));
        it=it+1;
   

    end
%     keyboard
%     cprop(:, end)=[];
    R = sum(weights(:, t+1:end),1);
    R = repmat(R, length(ind),1);
    R = R - weights(ind, t+1:end);
    gg=gvec(t+1:end);
    gg = repmat(gg, length(ind), 1);
    laccept = -2*gg.*R.*(wprop-wold)-gg.*(wprop.^2 - wold.^2);
    
    laccept = sum(laccept,2);
    

    u = rand(size(wprop, 1), 1);
    accept = exp(laccept);

    weights( ind(u<accept), t+1:end ) = wprop( u <accept, : );
    C( ind(u<accept), t:end ) = cprop( u <accept, : );

end


% For weights that first appear at time t
 
for t=2:T
    ind = find(sum(M(:, 1:t-1), 2) ==0);
    wold = weights(ind, 1:t-1);
    wmint = weights(ind, t);
    
    cold = C(ind, 1:t-1);

    cprop = zeros(size(cold));
    cprop(:, t-1)=  poissrnd(phi.*wmint);
    wprop = zeros(size(wold));
    it=t-1;
    
    for p = t-1:-1:1
        wprop(:, it) = (cprop(:, it)>0).*gamrnd(cprop(:, it), 1/(phi+tau));
        if it>1

        cprop(:, it-1)=  poissrnd(phi.*wprop(:, it));
        end
        it=it-1;
    end
   
    R = sum(weights(:, 1:t-1),1);
    R = repmat(R, length(ind),1);
    R = R - weights(ind, 1:t-1);
     
      gg=gvec(1:t-1);
    gg = repmat(gg, length(ind), 1);
    laccept = -2*gg.*R.*(wprop-wold)-gg.*(wprop.^2 - wold.^2);
    
    laccept = sum(laccept,2);
    

    u = rand(size(wprop, 1), 1);
    accept = exp(laccept);

    weights( ind(u<accept), 1:t-1 ) = wprop( u <accept, : );
    C( ind(u<accept), 1:t-1 ) = cprop( u <accept, : );
    

end
% 




end
