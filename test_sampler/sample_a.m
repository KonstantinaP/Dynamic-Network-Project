
function [alpha] = sample_a(weights, C, alpha, phi, tau, alpha_a, alpha_b)
[K, N] = size(weights);
alpha_std=0.1;
nb_MH = 1; % Nb of MH iterations
for nn=1:nb_MH
    
    alpha_new = alpha*exp(alpha_std*randn);
    
    u = rand;
    logaccept = sum((-alpha+alpha_new).* log((tau+phi).*weights(end, 2:N))...
        + gammaln(alpha+C(end, :))  - gammaln(alpha_new+C(end, :))) + (-alpha+alpha_new)*log(weights(end, 1)*(phi+tau)) ;
    prior_ratio = log(alpha_new) - log(alpha) + (alpha_a -1)*log(alpha_new) - (alpha_a-1)*log(alpha)...
            - alpha_b*(alpha_new) + alpha_b*alpha;
        
        logaccept = logaccept + prior_ratio;
        
    
    if rand<exp(logaccept) % If accept
        alpha = alpha_new;
    end
end