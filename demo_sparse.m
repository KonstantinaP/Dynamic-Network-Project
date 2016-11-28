%% BNPgraph package: demo_sparse
% 
% This Matlab script shows empirically the sparsity properties of a range of graph models.
%
% For downloading the package and information on installation, visit the
% <http://www.stats.ox.ac.uk/~caron/code/BNPgraph BNPgraph webpage>.
%
% Reference: F. Caron and E.B. Fox. <http://arxiv.org/abs/1401.1137 Sparse graphs using exchangeable random
% measures.>  arXiv:1401.1137, 2014.
%
% Author: <http://www.stats.ox.ac.uk/~caron/ François Caron>, University of Oxford
%
% Tested on Matlab R2014a.


%% General settings
%
close all
clear all

% Add paths
addpath('inference/utils/');

% Save plots and workspace
saveplots = false;
saveworkspace = true;
if saveplots
    rep = './plots/';
    if ~isdir(rep)
        mkdir(rep);
    end
end

% Set the seed
rng('default')

%% Definition of the graph models
% 


% Model 4: GGP (sigma=0)
alpha = 100; tau = 1; sigma = 0;
T= 5; rho=0.01; gvar = ones(1,T); settings.threshold=1e-3;
phi = 10;
field = 'alpha';
%  trial = 1:1:150;
%  trial = 1:0.5:130;
  trial = 1:0.5:210;
n_samples = 1;
settings.dt=1;
settings.gcontrol=0;
%% Sample graphs of various sizes
%

%     fprintf('--- Model %d/%d: %s ---\n', k, length(obj), obj{k}.name)    
    for i=1:length(trial) % Varying number of nodes
%         if rem(i, 100)==0
%             fprintf('Trial %d/%d \n', i, length(trial{k}));
%         end
i        
        alpha= trial(i);
        for j=1:n_samples % For different samples
            Z = dyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings); % Sample the graphs, symmetric.
            G = squeeze(Z(end, :, :)); % keep the last one in time
            nbnodes(i,j) = size(G, 1);
            nbedges(i,j) = sum(G(:))/2 + trace(G)/2;
            maxdegree(i,j) = max(sum(G));
            degreeone(i,j) = sum(sum(G)==1);
        end
    end


%% Some plots
%
save simulation_sparse_result nbnodes nbedges maxdegree degreeone
% Properties of the plots
plotstyles = {'--k', '--+b', '--xc', 'rs', 'rd', 'ro'};
colorstyle = {'k', 'b', [0,0,.6], [1,0,.5], 'r', [.6,0,0]};
leg = {'GGP (\sigma = 0)'};
set(0,'DefaultAxesFontSize', 12)
set(0,'DefaultTextFontSize', 16)

% Nb of Edges vs nb of nodes on loglog plot 
% pas = .4;
pas=0.1
k=5;
f1 = figure('name', 'edgesvsnodesloglog')
    h = plot_loglog(nbnodes(:), nbedges(:), plotstyles{k}, pas);
    set(h,'linewidth', 2, 'markersize', 8, 'markerfacecolor', colorstyle{k},'color', colorstyle{k});


xlabel('Number of nodes', 'fontsize', 16)
ylabel('Number of edges', 'fontsize', 16)
legend(leg,'fontsize', 16, 'location', 'northwest')
xlim([10, 500])
ylim([10, 20000])
legend('boxoff')
if saveplots
    saveas(f1, 'plots/edgesvsnodes.pdf');

%     savefigure(gcf, 'edgesvsnodes', rep);
end

%%
%

% Nb of Edges/Nb of nodes squared vs nb of nodes on loglog plot 
 pas = 1;
f2=figure('name', 'edgesvsnodesloglog')
    ind = nbnodes(:)>0;
    h = plot_loglog(nbnodes(ind), nbedges(ind)./nbnodes(ind).^2, plotstyles{k}, pas);
    set(h,'linewidth', 2, 'markersize', 8, 'markerfacecolor', colorstyle{k},'color', colorstyle{k});
    hold on

xlabel('Number of nodes', 'fontsize', 16)
ylabel('Nb of edges / (Nb of nodes)^2', 'fontsize', 16)
legend(leg,'fontsize', 16, 'location', 'southwest')
xlim([3, 700])
legend('boxoff')
if saveplots
    saveas(f2, 'plots/edgesvsnodes2.pdf');
%     savefigure(gcf, 'edgesvsnodes2', rep);
end

%%
%

% Nb of nodes of degree one versus number of nodes
f3=figure('name', 'degonevsnodes');
    h = plot_loglog(nbnodes(:), degreeone(:), plotstyles{k}, pas);
    set(h,'linewidth', 2, 'markersize', 8, 'markerfacecolor', colorstyle{k},'color', colorstyle{k});
    hold on
xlabel('Number of nodes', 'fontsize', 16);
ylabel('Number of nodes of degree one', 'fontsize', 16);
legend(leg,'fontsize', 16, 'location', 'northwest');
legend('boxoff');
xlim([10, 500])
ylim([1, 2000])
if saveplots
    saveas(f3, 'plots/degreeonevsnodes.pdf');
%     savefigure(gcf, 'degreeonevsnodes', rep);

end
% Save workspace
% if saveworkspace
%     save([rep 'test_stats']);
% end