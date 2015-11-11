% Test ynamic model
close all
clear all
K = 10;
T = 4;
% Test data


ID{1} = [1:K];
ID{2} = [1:K];
ID{3} = [1:K];
ID{4} = [1:K];
Z = cell(1, T);
% for t=1:T
%     r = rand(K, K);
%     r =  triu(r);
%     Z{t} = r>0.4;
%     
% end

% z=zeros(K, K);
% for k=1:2:K
%     z(k,:) = ones(1, K);
%
%     z(:, k) = ones(1, K);
%
% end

z=zeros(K, K);
z(1, :)= ones(1, K);
z(3, :)= ones(1, K);
 z=triu(z);
for t=1:T

    Z{t} = z;

end



% Run time-varying Plackett-Luce
alpha = 1;
N_Gibbs = 100;
N_burn = 0;
thin = 1;
phi = 1;%*ones(3, 19);
tau=1;
settings.typegraph = 'simple';
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
settings.times = 1:T;
settings.phi_a = .1;
settings.phi_b = .1;
settings.rho = 0; % decay rate. Controls the dependency on the interactions at the previous time step


settings.sample_correlation = 1;
[weights_st, alpha_st, phi_st, stats] = run_inference(Z, ID, alpha, tau, phi, N_Gibbs, N_burn, thin, settings);

