function [mnew_res, mold_res, mn] = update_interaction_counts(t, Zt, logw, Nnew, Nold, ind1, ind2, N, settings)
% Metropolis- Hastings. Jointly update mnew and mold



mnew_t = Nnew(:, :, t);

if t == 1
    mnew_mt = zeros(size(mnew_t));
    deltat=0;
else
    mnew_mt = Nnew(:, :,t-1);
    deltat = settings.times(t)-settings.times(t-1);

end

if t==N
    mold_plust = zeros(size(Nold(:, :, t)));

else
    mold_plust = Nold(:, :, t+1);

end
    
mold_t = Nold(:, :, t);
pi = exp(-settings.rho*deltat);

id = find(Zt);% Zt should be in the upper triangular form

mold_prop = binornd(mnew_mt(id), pi.*ones(length(id),1));

[k]= length(logw);
lograte_poi = log(2) + logw(ind1) + logw(ind2);
lograte_poi(ind1==ind2) = 2*logw(ind1(ind1==ind2));
nd = poissrnd(exp(lograte_poi));
td = tpoissrnd(exp(lograte_poi));
%count = sparse(ind1, ind2, d, k, k);

mnew = mnew_t(id);
mold = mold_t(id);
mold_plt = mold_plust(id);
mnew_prop = td.*(mold_prop==0) + nd.*(mold_prop>0);

logaccept = gammaln(mnew_prop+1) +gammaln(mnew +mold_plt +1) - gammaln(mnew+1)-gammaln(mnew_prop-mold_plt+1)+(mnew_prop - mnew)*log(1-pi);
 

u = rand(length(id),1);
    
    accept = exp(logaccept);
    mnew(u<accept) = mnew_prop(u<accept);
    mold(u<accept) = mold_prop(u<accept);
   
    mnew_res= zeros(k, k);
    mold_res= zeros(k, k);
    
     mnew_res(sub2ind(size(mnew_res), ind1', ind2')) = mnew;
    mold_res(sub2ind(size(mnew_res), ind1', ind2')) = mold;
mn = sum(mnew_res,1)' + sum(mnew_res, 2);
end