function [weights C] = sample_weights_add(C, weights, M, phi, tau)

% Function that jointly samples the c_{tk} and w_{t+1k}
% M - KxT

[K, N] = size(weights);
nb_MH = 1; % Nb of MH iterations
counts = [C(1:K-1, :) zeros(K-1, 1)]; % padded with zero, needed for the acceptance ratio ratio

R = sum(weights,1);
R= repmat(R, K-1,1);
true_weights=weights;
true_c = C;
weights= weights(1:K-1, :);
wac = weights(:, 2:N);
cac = C(1:K-1, :);
R = R - weights; % K x N

for nn=1:nb_MH
    Ctnew = poissrnd(phi.*(weights(:, 1:N-1)));
    
    Wttnew= (Ctnew>0).*gamrnd(counts(:, 1:N-1), 1/(phi+tau)); % the 2:N part of the weights
    
    
    temp = Wttnew.^2 - weights(:, 2:N).^2 +(2*R(:, 2:N) +phi).*(Wttnew - weights(:, 2:N));
   logaccept= M(:, 2:N).*log(Wttnew - weights(:, 2:N)) + (Ctnew - counts(:, 1:N-1) + counts(:, 2:N)).*log(Wttnew)...
        + (Ctnew - counts(:, 1:N-1) - counts(:, 2:N)).*log(weights(:, 2:N))+2*(counts(:, 1:N-1) - Ctnew) ...
        - temp;
    
    u = rand(K-1, N-1);
    
    accept = exp(logaccept);
    cac(u<accept) = Ctnew(u<accept);
     wac(u<accept) = Wttnew(u<accept);
    true_weights(1:K-1, 2:N) = wac;
    true_c(1:K-1, :) = cac;
    
end
weights = true_weights;
C = true_c;


end
