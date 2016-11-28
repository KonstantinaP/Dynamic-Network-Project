function [mnew_res, mold_res, mn] = update_interaction_counts(t, Z, w, Nnew, Nold, ind, N, settings)
% Metropolis- Hastings. Jointly update mnew and mold
T = size(Nold,3);

% for t=1:T
    
    mnew_t = Nnew(:, :, t);
    mold_t = Nold(:, :, t);
    
    
    if t == 1
        mnew_mt = zeros(size(mnew_t));
        mold_mt = zeros(size(mold_t));
    else
        mnew_mt = Nnew(:, :,t-1);
        mold_mt = Nold(:, :,t-1);
        
        
    end
            

    if t==N
        mold_plust = zeros(size(Nold(:, :, t)));
        deltat=0;
        
    else
        mold_plust = Nold(:, :, t+1);
        deltat = settings.times(t+1)-settings.times(t);
        
    end
    
    pi = exp(-settings.rho*deltat);
    
    id = find(Z{t});% Zt should be in the upper triangular form
    
    mold_prop = binornd(mnew_mt(id) + mold_mt(id), pi.*ones(length(id),1));
        [k]= size(Nnew(:,:,t),2);

    logw=w(1:k,t);
    
    ids=ind{t};
ind1=ids(:,1);
ind2=ids(:,2);

    lograte_poi = log(2) + logw(ind1) + logw(ind2);
    lograte_poi(ind1==ind2) = 2*logw(ind1(ind1==ind2));
    nd = poissrnd(exp(lograte_poi));
    td = tpoissrnd(exp(lograte_poi));
    %count = sparse(ind1, ind2, d, k, k);
    
    mnew = mnew_t(id);
    mold = mold_t(id);
    mold_plt = mold_plust(id);
    mnew_prop = poissrnd(exp(lograte_poi));
%mnew_prop(mold_prop==0) = tpoissrnd(exp(lograte_poi(mold_prop==0)));


mnew_prop = td.*(mold_prop==0) + nd.*(mold_prop>0);



    
    proposal = mnew_prop + mold_prop;
    current = mnew + mold;
    
    logaccept = gammaln(proposal+1) +gammaln(current +mold_plt +1) - gammaln(current+1)-gammaln(proposal-mold_plt+1)+(proposal - current)*log(1-pi);
    
    
    u = rand(length(id),1);
    
    accept = exp(logaccept);
    
    if t==N
        accept=1;
    end
    mnew(u<accept) = mnew_prop(u<accept);
    mold(u<accept) = mold_prop(u<accept);
    
    mnew_res= zeros(k, k);
    mold_res= zeros(k, k);
    
   % mnew_res(sub2ind(size(mnew_res), ind1', ind2')) = mnew;
  
    mnew_res(id) = mnew;
    
    
%     mold_res(sub2ind(size(mnew_res), ind1', ind2')) = mold;
mold_res(id) = mold;
    mn = sum(mnew_res,1)' + sum(mnew_res, 2);
    
   % Nnew(:,:,t)= mnew_res;
   % Nold(:,:,t)= mold_res;
   % M(:, t)= mn;
   
% end
end