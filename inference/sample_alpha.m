% Metropolis-Hastings for sampling hypeparameter alpha

function [alpha Wst Cst] = sample_alpha(weights, C, alpha, phi, tau, alpha_a, alpha_b)


[K, N] = size(weights);
alpha_std=0.1;
nb_MH = 1; % Nb of MH iterations
for nn=1:nb_MH
    
%     alpha_new = alpha*exp(alpha_std*randn);
%     
%     u = rand;
%     logaccept = sum((-alpha+alpha_new).* log((tau+phi).*weights(end, 2:N))...
%         + gammaln(alpha+C(end, :))  - gammaln(alpha_new+C(end, :))) + (-alpha+alpha_new)*log(tau*weights(end, 1)) + gammaln(alpha) - gammaln(alpha_new) ;
%     
%     if rand<exp(logaccept) % If accept
%         alpha = alpha_new;
%     end
%     
    
 [alpha] = slice_sample_alpha(weights, C, alpha, phi, tau, alpha_a, alpha_b);
 
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% Update w_{t\ast}, c_{t\ast}
    
    Wst = weights(end, :);
    Cst = C(end, :);
    
    wsnew = gamrnd(alpha, 1/tau);
    logaccept = - (sum(weights(:,1))+wsnew)^2 + (sum(weights(:,1))+weights(end, 1))^2 - phi*(wsnew - weights(end, 1))...
        + C(end, 1)*log(wsnew)-log(weights(end,1));
    
    if rand<exp(logaccept)
        Wst(1)  = wsnew;
    end
    
    for t=2:N
        nb_MH = 1; % Number of MH iterations
        for nn=1:nb_MH
            
            %%%% Update c_{t\ast}
            csnew = poissrnd(phi.*(Wst(t-1)));
            
            
            logaccept = (-C(end, t-1)+csnew).* log((tau+phi).*weights(end, t))...
                + gammaln(alpha+C(end,t-1))  - gammaln(alpha+csnew);
            
            
            if rand<exp(logaccept)
                Cst(t-1)  = csnew;
            end
            
            %%%% Update w_{t\ast}
            wsnew = gamrnd(alpha +(t>1)*Cst(t-1), 1/(phi*(t>1)+tau));
            cc=0;
            if t<N
                cc = C(end, t);
            end
            logaccept = - (sum(weights(:,t))+wsnew)^2 + (sum(weights(:,t))+weights(end, t))^2 ...
                - phi*(wsnew - weights(end, t))...
                + cc*(log(wsnew)-log(weights(end,t)));
            
            if rand<exp(logaccept)
                Wst(t)  = wsnew;
            end
            
        end
    end
    
end