function [tau ] = sample_tau(tau, tau_a, tau_b, weights, alpha, phi, counts)



%  note here that the last row of weights(end, :) contains the values of
% $w_{t\ast}$

[K, N] = size(weights);
a = zeros(K, N-1);
a(end, :) = alpha;

tau_std =.1;
nb_MH = 1;

mask = (weights(:, 2:N)~= 0)& counts(:, 1:N-1)==0 ;
mask=logical([zeros(K, 1) mask]);
% add this to the wrem
%
%weights(end, :) =  weights(end, :) + sum(weights.*mask, 1);


for nn=1:nb_MH % number of MH iterations
    % Sample phi w.r.t. weights (marginalised over C )
    
    tau_new = tau*exp(tau_std*randn);
    
    lograte = (a + counts(:, 1:N-1)).*(log(tau_new+phi) - log(tau+phi)) ...
              -(tau_new-tau).*weights(:, 2:N);
%           keyboard
%     
%    lograte(mask(:, 2:end))=0;
%     
   
   
%     lograte
%     keyboard
%     
    wfall = sum(weights(:,1));
    lograte =  sum(sum(lograte))...      
                + (tau_a)*(log(tau_new) - log(tau))...
                - tau_b*(tau_new-tau); ...
                + (alpha)*(log(tau_new) - log(tau))...
                - wfall*(tau_new-tau);
    

    if rand<exp(lograte) % If accept
        
        tau = tau_new;
    end
    
end