% Metropolis-Hastings for sampling W_st

function [weights] = ggp_sample_Wst(weights, C, alpha, sigma, tau, phi, gvar)

[K, N] = size(weights);
nb_MH = 1; % Nb of MH iterations



for nn=1:nb_MH
    sum_w = sum(weights(1:end-1, :));
    Wst = weights(end, :);
    Cst = C(end, :);
    
    co = [ 0 Cst(1:N-1)];
    cc = co + Cst;
    a= cc;
    
    b = 2*phi + 2*sum_w+ Wst;
    wsnew = gtGGPsumrnd(alpha, sigma, tau, a, b);
    logaccept = -((sum_w + wsnew).^2)+((sum_w + Wst).^2) + ...
                + ( 2*sum_w + Wst).*wsnew - (2*sum_w + wsnew).*Wst; 
     u=rand(size(logaccept))<exp(logaccept);
    weights(end, u) =  wsnew(u);
    
    %%%%
  
    
end





end


