function [mnew_res, mold_res, mn] = ggp_sample_interaction_counts(t, Zt, logw, Nnew, Nold, ind1, ind2, N, gv, rho, settings)
% Metropolis- Hastings. Jointly update mnew and mold
% all output matrices are in upper triangular form.

k = size(logw,1);
mnew_t = Nnew{t};
mold_t = Nold{t};

deltat = settings.dt;

if t<N
    
    if t == 1
        mnew_mt = sparse(k, k);
        mold_mt = sparse(k, k);
    else
        mnew_mt = Nnew{t-1};
        mold_mt = Nold{t-1};
        
        
    end
    
    
    mold_plust = Nold{t+1};
    
    
    
    pi = exp(-rho*deltat);
    if pi==0
        pi =0.9999;
    end
    
    id = find(Zt);% Zt should be in the upper triangular form
    
    % keyboard
    mpt = mnew_mt(id) + mold_mt(id);
    mold_prop = binornd(mpt, pi.*ones(length(id),1));
    
    % keyboard
    lograte_poi = log(2) + log(gv)+logw(ind1) + logw(ind2);
    lograte_poi(ind1==ind2) = log(gv)+2*logw(ind1(ind1==ind2));
    
    threshold =0;% mold_plust(id)-mold_prop ;
    nd = threshold +poissrnd(exp(lograte_poi));
    td = threshold + tpoissrnd(exp(lograte_poi));
    %count = sparse(ind1, ind2, d, k, k);
    
    mnew = full(mnew_t(id));
    mold = full(mold_t(id));
    mold_plt = full(mold_plust(id));
    mnew_prop = td.*(mold_prop==0) + nd.*(mold_prop>0);
    
    
    proposal = mnew_prop + mold_prop;
    current = mnew + mold;
    
    if pi==0
        pi =0.9999;
    end
aa=current - mold_plt +1;
bb=proposal-mold_plt+1;
aa(aa<0)=NaN;
bb(bb<0)=NaN;
    logaccept = gammaln(proposal+1) +gammaln(aa) - gammaln(current+1)-gammaln(bb)+(proposal - current)*log(1-pi);
  
    u = rand(length(id),1);
%     t
    accept = exp(logaccept);
%     keyboard
    
    mnew(u<accept) = mnew_prop(u<accept);
    mold(u<accept) = mold_prop(u<accept);
    
else
    if N==1
        mnew_mt = sparse(k, k);
        mold_mt = sparse(k, k);
        else
    mnew_mt = Nnew{t-1};
    
        mold_mt = Nold{t-1};
    end
        
    pi = exp(-rho*deltat);
    
    id = find(Zt);% Zt should be in the upper triangular form
    
    % keyboard
    mold= binornd(mnew_mt(id) + mold_mt(id), pi.*ones(length(id),1));
    
    % keyboard
    lograte_poi = log(2) + log(gv)+logw(ind1) + logw(ind2);
    lograte_poi(ind1==ind2) = log(gv)+2*logw(ind1(ind1==ind2));
    nd = poissrnd(exp(lograte_poi));
    td = tpoissrnd(exp(lograte_poi));
    mnew =td.*(mold==0) + nd.*(mold>0);
    
end
%    mnew_res= sparse(k, k);
%   mold_res= sparse(k, k);
mnew_res = sparse(ind1, ind2, mnew, k, k);
mold_res = sparse(ind1, ind2, mold, k, k);

%      mnew_res(sub2ind(size(mnew_res), ind1', ind2')) = mnew;
%     mold_res(sub2ind(size(mnew_res), ind1', ind2')) = mold;

mn =full (sum(mnew_res,1)' + sum(mnew_res, 2));

if sum(mn<0)
    keyboard
end


end
