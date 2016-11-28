function [gvar ] = sample_gvar_mh(weights, nnew, alpha, phi, tau, gvar, g_a, g_b)


[K T] = size(weights);
k=K-1;
dg = k*(k-1)/2;
nmtx = sparse(T, dg);
% 
% for t=1:T
%     temp = triu(2*weights(1:end-1, t)*weights(1:end-1,t)',1);
%     wmtx(t,:) = temp(triu(true(k,k) ,1));
%     
%    nmtx(t, :) = nnew{t}(triu(true(k, k), 1));
% end
%  
% 
% lambda = .5;
% gprop = gvar.*exp(lambda*randn(1, T));
% gmtx = repmat((gprop - gvar)', 1, dg);
% lgmtx = repmat((log(gprop) - log(gvar))', 1, dg);
% 
% 
% logaccept =sum(sum( nmtx.*lgmtx -wmtx.*gmtx));
% logaccept = logaccept -g_b.*(gprop - gvar) + g_a.*(log(gprop) - log(gvar));
% 
% u = rand(1, T);
% 
% gvar(u<exp(logaccept)) = gprop(u<exp(logaccept));

  lambda = .1;

for t=1:T
    gcur = gvar(t);
gprop = gvar(t).*exp(lambda*randn);  
w = weights(1:end-1,t);
wmtx = sparse((w*w').*(triu(ones(k, k),1)));
logaccept =sum(sum( nnew{t}.*(log(gprop) - log(gcur))- 2.*wmtx.*(gprop - gcur) ) );
  logaccept = logaccept -g_b.*(gprop - gcur) + g_a.*(log(gprop) - log(gcur)); 
  
 if rand<exp(logaccept)
     gvar(t) = gprop;
 end 


end

end