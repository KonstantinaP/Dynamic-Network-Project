% Test dynamic model
addpath '../'
addpath '../inference/utils/'
close all
clear all

seed=4;
rand('seed', seed);
randn('seed', seed);
%% Generate toy data
% for t=1:T
%     ID{t} = [1:K];
% end
% % ID{3} = [1:K];
% % ID{4} = [1:K];
% % Z = cell(1, T);
% % % for t=1:T
% %     r = rand(K, K);
% %     r =  triu(r);
% %     Z{t} = r>0.4;
% %
% % end
% 
% z=zeros(K, K);
% for k=1:2:K
%     z(k,:) = ones(1, K);
%     
%     %z(:, k) = ones(1, K);
%     
% end
% 
% % z=zeros(K, K);
% % z(1, :)= ones(1, K);
% % z(3, :)= ones(1, K);
% 
% 
% z=triu(z);
% for t=1:T
%     
%     Z(t, :,:) = z;
%     
% end

%% OR generate from the process
true_alpha=5;
true_sigma =0;
true_tau =1;
T=1;
true_phi= 100;
settings.rho =0.1;
settings.dt=1;


settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-4;

[G, true_w, true_c, true_K, true_Nnew, true_Nold, true_N, true_M, indchainmT, indlinks] = dyngraphrnd(true_alpha, true_sigma, true_tau, T, true_phi, settings);

true_w=true_w(:, 1:true_K(T));
indlog = false(1, true_K(T)) ;
indlog(indlinks) = true;
true_wrem =  sum(true_w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
true_w=true_w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
true_w = [true_w'; true_wrem' ];

true_crem =  sum(true_c(:, ~indlog),2);
true_c=[true_c(:, indlog)';  true_crem'];
true_M =true_M(indlinks, :);



%[Z, weights, c, K, Knew, N_new, N_old, N, M, indchain, indlinks]= dyngraphrnd(alpha, sigma, tau, T, phi, rho, dt, settings);


ID = cell(1, T);
for t=1:T

    Z{t} = squeeze(G(t, indlinks, indlinks));
end


%%%%%%%%%%%%%%%%%%%%%%% RUN THE MODEL
% keyboard
alpha = 5;
N_Gibbs = 5000;
N_burn = 0;
thin = 1;
phi = 100;%*ones(3, 19);
tau=1;
settings.typegraph = 'simple';
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
settings.times = 1:T;
settings.phi_a = .1;
settings.phi_b = .1;
settings.alpha_a = .1;
settings.alpha_b = .1;
settings.rho = 0.1; % decay rate. Controls the dependency on the interactions at the previous time step

settings.sample_phi=0;
settings.sample_alpha=0;
[weights_st, alpha_st, phi_st, stats, weights, C, nnew, nold] = run_inference(Z, alpha, tau, phi, N_Gibbs, N_burn, thin, settings);


% Trace plots and posterior histograms of alpha, sigma, tau and w_*
col = {'k', 'r', 'b'};
variables = {'alpha', 'phi',  'w_rem'};
namesvar = {'\alpha', '\phi', 'w_*'};
truevalues = {true_alpha, true_phi, true_wrem};
% for i=1:1
%     figure
%    % for k=1:nchains
%         plot(thin:thin:N_Gibbs, alpha_st, '-r');
%         hold on
%     %end
%     plot(thin:thin:N_Gibbs, true_alpha*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
%     legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
%     legend boxoff
%     xlabel('MCMC iterations', 'fontsize', 16);
%     ylabel(namesvar{1}, 'fontsize', 16);
%     box off
%     xlim([0, N_Gibbs])
% 
%     figure
%     hist(alpha_st, 30)
%     hold on
%     plot(truevalues{1}, 0, 'dg', 'markerfacecolor', 'g')
%     xlabel(namesvar{1}, 'fontsize', 16);
%     ylabel('Number of MCMC samples', 'fontsize', 16);
% end

% for i=1:1
%     figure
%    % for k=1:nchains
%         plot(thin:thin:N_Gibbs, phi_st, '-r');
%         hold on
%     %end
%     plot(thin:thin:N_Gibbs, true_phi*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
%     legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
%     legend boxoff
%     xlabel('MCMC iterations', 'fontsize', 16);
%     ylabel(namesvar{2}, 'fontsize', 16);
%     box off
%     xlim([0, N_Gibbs])
% 
%     figure
%     hist(phi_st, 30)
%     hold on
%     plot(truevalues{2}, 0, 'dg', 'markerfacecolor', 'g')
%     xlabel(namesvar{2}, 'fontsize', 16);
%     ylabel('Number of MCMC samples', 'fontsize', 16);
% end


for i=1:T
    figure
   % for k=1:nchains
        plot(thin:thin:N_Gibbs,  sum(weights_st{i},2), '-r');
        hold on
    %end
    plot(thin:thin:N_Gibbs, true_wrem(i)*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
    legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
    legend boxoff
    xlabel('MCMC iterations', 'fontsize', 16);
    ylabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
    box off
    xlim([0, N_Gibbs])

    figure
    hist(sum(weights_st{i},2), 30)
    hold on
    plot(true_wrem(i), 0, 'dg', 'markerfacecolor', 'g')
    xlabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
    ylabel('Number of MCMC samples', 'fontsize', 16);
end



% Credible intervals for the weights
kk =size(true_w,1);
for t=1:T
    nZ = Z{t};
%     nZ=nZ(1:kk, 1:kk);
[~, indt{t}] = sort(sum(nZ), 'descend');

end
for t=1:T
 figure
   
for k=1:min(kk-1, 50)
         plot([k, k],...
            quantile(weights_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
            'linewidth', 3);
        hold on
        plot(k, true_w(indt{t}(k), t), 'xg', 'linewidth', 2)       
end
xlim([0.1,min(true_K(T), 50)+.5])
legend('95% credible intervals', 'True value')
legend boxoff
box off
xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
ylabel('Sociability parameter', 'fontsize', 16)


end