% test_dyngraphrnd
close all
clear all


% Add paths
addpath('../../inference/utils/');


% Set the seed
rng('default')
% 
% seed=2;
% rand('seed',seed);
% randn('seed',seed);

alpha = 3; sigma = 0.8; tau = 1; % Parameters gamma process
phi = 100; % tunes dependence in dependent gamma process 
rho = 1; % death rate for latent interactions
T = 4; % Number of time steps
settings.dt=1;
gvar = ones(1,T); 
settings.threshold=1e-3;
settings.gcontrol=0;
[Z, w, c, K, N_new, N_old, N, M, indlinks] = GGPdyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings);

figure 
weights = w(:, 1:K(T));
plot(weights)

keyboard
% figure
% plot(K)

figure
imagesc(c)
% 
figure
for t=1:T
    subplot(2,T/2,t)
    imagesc(squeeze(N(t,:,:)));
    title(['t=' num2str(t)])
end

figure
for t=1:T
    subplot(2,T/2,t)
    imagesc(squeeze(Z(t,:,:)));
    title(['t=' num2str(t)])
end

filename = 'test.gexf';
exportgexf(Z, filename);

% Settings Gephi:
% Enable timeline
% Choose note ranking degree click on continously auto-apply click on button bottom left for local update
% Play settings: delay 200ms, stepsize 0.5%
% Layout Fruchterman area 10000, gravity 30.0, speed 1.0

% 
% 
% for t=1:T
%     mtx{1} = Z(t, :,:);
%     active_n = find(active_nodes(t,:)>0);
%     %you can first define the number of nodes in the network:
%     
%     N = size(G{t},1);
%     %and then a corresponding set of coordinates on the perimeter of a circle:
%     f3 = figure
%     coords = [cos(2*pi*(1:N)/N); sin(2*pi*(1:N)/N)]';
%     %Then you can plot the graph using gplot:
%     mtx{2} =active_n;
%     my_gplot(mtx, coords);
%     %and mark the vertices using text:
%     
%     text(coords(:,1) - 0.1, coords(:,2) - 0.05, num2str((1:N)'), 'FontSize', 14)
%     set(gca,'YTick',[])
%     set(gca,'XTick',[])
%     title(['t = ' num2str(times(t))]);
% %     if save_to_file
% %         saveas(f3, ['figures/graph_timeloc_' num2str(times(t)) '.pdf']);
% %     end
% %     
% end
