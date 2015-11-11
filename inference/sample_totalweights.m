% Function that samples the total weighs from the prior to 
% improve mixing. The individual weights are accordingly rescaled. 


function weights = sample_totalweights(weights, phi, alpha, tau)
[K, N] = size(weights);

S = zeros(N, 1);
S(1) = gamrnd(alpha, 1/tau);
indnotNaN = isnan(weights(:, 1)) ==0;
weights(:, 1) = S(1)*weights(:, 1)/sum(weights(indnotNaN, 1));
for t=2:N
    % Use a normal approximation to the poisson distribution to fasten
    % code
    temp = poissrnd(phi*S(t-1));
%         temp = max(0, normrnd(phi(1, t-1)*S(t-1), sqrt(phi(1, t-1)*S(t-1))));
    S(t) = gamrnd(alpha + temp, 1/(tau + phi));
    
    indnotNaN = isnan(weights(:, t)) ==0;
    % Rescale the weightss
    weights(:, t) = weights(:, t)*S(t)/sum(weights(indnotNaN, t));  
end   