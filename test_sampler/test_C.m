clear all
close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
% create the Pitt-Walker model 

% 
% seed=10;
% randn('seed', seed);
% rand('seed', seed);


%% Sample the latent C_1,C_2,...
tau=1; 
alpha=5; 
T=10;
phi=1;

indmax = 1000; % so that to allocate memory
c = zeros(T, indmax);
w = zeros(T, indmax);


w1_all = gamrnd(alpha, 1/tau);
c1_all = poissrnd(phi*w1_all);
if c1_all>0
    [~, m, K(1)] = crprnd(alpha, c1_all);
    for i=1:K(1)
        c(1, i) = m(i);
        theta(i) = alpha*rand;
    end
else
    K(1) = 0;
end

wrest = zeros(1, T);
crest = zeros(1, T);
for t=2:T
    % Sample counts for existing atoms
    ind = c(t-1,:)>0;
    w(t, ind) = gamrnd(c(t-1,ind), 1/(tau+phi));
    c(t,ind) = poissrnd(phi*w(t, ind));
    % Sample counts for new atoms
    w_rest = gamrnd(alpha, 1/(tau+phi));
    c_rest = poissrnd(phi * w_rest);
    if c_rest>0
        [~, m, K_new] = crprnd(alpha, c_rest);    
        for i=1:K_new
            c(t, K(t-1)+i) = m(i);
            theta(K(t-1)+i) = alpha*rand;
        end
        K(t) = K(t-1) + K_new;   
    else
        K(t) = K(t-1);
    end
    wrest(t) = w_rest;
    crest(t) = c_rest;
end
trueC = c(2:T-1, 1:K(T-1))';
trueC(end+1,:) = crest(2:end-1);

weights = w(2:T, 1:K(T-1))';
weights(end+1, :) = wrest(2:end);
% % sample from GGP in the local steps
% local_w = zeros(T, indmax);
% kmax=0;
% sigma=0;
% for t=1:T
%     
%     
%     [temp, threshold] = GGPrnd(alpha, sigma, tau)
%     local_w(t, 1:length(temp))=temp;
%     kmax= max(kmax, length(temp));
%     
%    
% end
% 
% local_w = local_w(:, 1:kmax);
% m

% 


% Init C
iters=5000;
C = poissrnd(phi.*(weights(:, 1:end-1)));
C(isnan(weights))=0;
c_samples =  cell(1, iters);

for t=1:iters
    
    C = sample_C(C, weights, phi, alpha, tau);
    
    c_samples{t} = C;
end
