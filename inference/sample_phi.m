function phi = sample_phi(phi, phi_a, phi_b, weights, alpha, tau)   



%  note here that the last row of weights(end, :) contains the values of
% $w_{t\ast}$

[K, N] = size(weights);
a = zeros(K, N-1);
a(end, :) = alpha;

phi_std = 1; 
nb_MH = 1;
for nn=1:nb_MH % number of MH iterations
    % Sample phi w.r.t. weights (marginalised over C )
        phi_new = phi*exp(phi_std*randn);
        
        temp = 2*sqrt((tau+phi).*phi.*weights(:, 1:N-1).*weights(:, 2:N));
        temp_new = 2*sqrt((tau+phi_new).*phi_new.*weights(:, 1:N-1).*weights(:, 2:N));
        
        lograte = (phi_new - phi) .*(weights(:, 1:N-1) + weights(:, 2:N))...
            + (a+1)/2.*log(tau+phi) -(a-1)/2 .*log(phi) ...
             - (a+1)/2.*log(tau+phi_new) +(a-1)/2 .*log(phi_new)...
             +log(besseli(a-1,temp,1))...
             -log(besseli(a-1,temp_new, 1)) ...
             + temp - temp_new;  % correction for using besseli(:,:,1) (see matlab definition of besseli)

         
         lograte(isnan(lograte)) = 0;
         
         % If weights(k, t+1)=0, different likelihood
         mask0 = (weights(:, 2:N)==0);
         lograte(mask0) = (phi_new - phi) .* weights(mask0);
%          
%          lograte = - sum(sum(lograte))...
%             + log(phi_new) - log(phi)...
%             + log(gampdf(phi_new, phi_a, 1/phi_b))...
%             - log(gampdf(phi, phi_a, 1/phi_b));                          

    
         lograte = - sum(sum(lograte))...
            + log(phi_new) - log(phi)...
            + (phi_a -1)*log(phi_new) - (phi_a-1)*log(phi)...
            - phi_b*(phi_new) + phi_b*phi;

        if rand<exp(lograte) % If accept
            phi = phi_new;
        end
        if phi==0
            keyboard
        end
%     end
end