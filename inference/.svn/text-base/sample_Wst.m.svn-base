% Metropolis-Hastings for sampling W_st

function [weights] = sample_Wst(weights, C, alpha, phi, tau, gvar)

[K, N] = size(weights);
nb_MH = 1; % Nb of MH iterations



for nn=1:nb_MH
    sum_w = sum(weights(1:end-1, :));
    Wst = weights(end, :);
    Cst = C(end, :);
    
    fi = (tau +2*phi).*ones(1, N);
    fi(1) = tau+ phi;
    co = [ 0 Cst(1:N-1)];
    cc = co + Cst;
%     fi=0;
    wsnew = GGPsumrnd(alpha + cc, 0, fi + 2*(gvar.*sum_w) + gvar.*Wst); %linearization    
    
    logaccept =  -gvar.*((sum_w + wsnew).^2)+gvar.*((sum_w + Wst).^2) + ...
        + gvar.*( 2*sum_w + Wst).*wsnew - gvar.*(2*sum_w + wsnew).*Wst +...
        + (alpha + cc).*(log(fi + gvar.*(2*sum_w +wsnew)) -log(fi + gvar.*(2*sum_w +Wst)));   
    
    
 
    
    u=rand(size(logaccept))<exp(logaccept);
    weights(end, u) =  wsnew(u);
    
end





end