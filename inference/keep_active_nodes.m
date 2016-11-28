function [S, N_new, N_old, N, M, wout, cout, wrem, ind] = keep_active_nodes(w, c, indchain, Z, N_new, N_old, N)

maxclust = size(Z(1, :,:));

ind = sum(squeeze(sum(Z,1)),1)>0;
indlog = false(size(ones(1,maxclust)));
indlog(ind) = true;
wrem =  sum(w(:, ~indlog),2);
wout =w(:, indlog);




% Keep the graphs with the active nodes
S = Z(:, ind, :);
S = S(:, :, ind);

N_new= N_new(:, ind,:);
N_new = N_new(:, :, ind);
N_old= N_old(:, ind,:);
N_old = N_old(:, :, ind);
N= N(:, ind,:);
N = N(:, :, ind);


cout=c;

for t=1:size(Z,1);
    M(:, t) = sum(squeeze(N_new(t, :, :)),1)' + sum(squeeze(N_new(t, :, :)), 2) - diag(squeeze(N_new(t, :, :)));    
end

end