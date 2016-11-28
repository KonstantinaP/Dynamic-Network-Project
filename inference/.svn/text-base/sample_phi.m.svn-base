function [phi ] = sample_phi(phi, phi_a, phi_b, weights, alpha, tau)



%  note here that the last row of weights(end, :) contains the values of
% $w_{t\ast}$

[K, N] = size(weights);
a = zeros(K, N-1);
a(end, :) = alpha;

phi_std =.1;
nb_MH = 1;
%
% phi = ones(K, N-1).*phi;
%         phi_new = ones(K, N-1).*phi;


mask = (weights(:, 2:N)~= 0)& weights(:, 1:N-1)==0 ;
mask=logical([zeros(K, 1) mask]);
% add this to the wrem
%
weights(end, :) =  weights(end, :) + sum(weights.*mask, 1);


for nn=1:nb_MH % number of MH iterations
    % Sample phi w.r.t. weights (marginalised over C )
    
    phi_new = phi*exp(phi_std*randn);
    
    temp = 2*sqrt((tau+phi).*phi.*weights(:, 1:N-1).*weights(:, 2:N));
    temp_new = 2*sqrt((tau+phi_new).*phi_new.*weights(:, 1:N-1).*weights(:, 2:N));
    
    lograte = -(phi_new - phi) .*(weights(:, 1:N-1) + weights(:, 2:N))...
        - ((a+1)./2).*log(tau+phi) - ((1-a)./2) .*log(phi) ...
        + ((a+1)./2).*log(tau+phi_new) +  ((1-a)./2) .*log(phi_new)...
        -log(besseli(a-1,temp,1))...
        +log(besseli(a-1,temp_new, 1)) ...
        - abs(real(temp)) +  abs(real(temp_new));  % correction for using besseli(:,:,1) (see matlab definition of besseli)
%     keyboard
     lograte(isnan(lograte)) = 0;
    lograte(mask(:, 2:end))=0;
    
    % If weights(k, t+1)=0, different likelihood
    mask0 = (weights(:, 2:N)==0);
    %          lograte(mask0) = (phi_new(mask0) - phi(mask0)) .* weights(mask0);
    lograte(mask0) = -(phi_new - phi) .* weights(mask0);

    %
             lograte = sum(sum(lograte))...
                + (phi_a)*(log(phi_new) - log(phi))...
                - phi_b*(phi_new-phi);
    % addterm
    % lograte =  sum(sum(lograte))...
    %             + log(unique(phi_new(phi>0))) - log(unique(phi(phi>0)))...
    %             + log(gampdf(unique(phi_new(phi>0)), phi_a, 1/phi_b))...
    %             - log(gampdf(unique(phi(phi>0)), phi_a, 1/phi_b))+addterm;
    %
    %
    
    % phi = unique(phi);
    if rand<exp(lograte) % If accept
        phi = unique(phi_new);
    end
    
end