function [trZ, weights, nlnks, ind] = init_model(Z, alpha, tag, settings)
% INPUT
% Z: cell of length N. Each element is a binary matrix of the links between
% K nodes. May or may not be triangular. 
%
% tag: '1' indicates that the weights do not have to appear in the
%   likelihood before first appearing in the data.


% OUTPUT
% weights: K+1 x N matrix of the sociabilities
% n      : vector of size N. total number of nodes at each timestep.
% ind    : cell structure of length N. It contains the indices of the links
%        i.e. Z_{tij}>0  at each time step.

% trZ: triangular Z. (always triangular the output graph)

N = length(Z); % N = total # of time steps
K = size(Z{1}, 1);


% Work on the graphs a bit...
if strcmp(settings.typegraph, 'simple')
    issimple = true;
else
    issimple =false;
end

ind=cell(1, N);
%nind=cell(1, N);

for t=1:N
    G= Z{t};
   
    if issimple % If no self-loops
        G2 = triu((G+G')>0, 1);% G2 upper triangular
        temp = tril(ones(size(G2)));
    else
        G2 = triu((G+G')>0);
        temp = tril(ones(size(G2)), -1);
    end
     ids = find(sum(G2,1));
    [ind1, ind2] = find(G2); % the indices refer to the upper triangle

    ind{t} =[ind1 ind2];
    
    
   % [nind1, nind2] = find(temp + G2 == 0);
   % nind{t} = [ids(nind1)' ids(nind2)'];
    trZ{t} = G2;
end



% trZ should be in UPPER triangular form!

nlnks = zeros(K, N);
for t=1:N
    nlnks(:, t) = sum(trZ{t}, 2);
end
nlnks(K+1, : ) = alpha;

weights = ones(K+1, N);
if tag
    for k=1:K+1
        first_time = find(nlnks(k, :), 1);
        weights(k, 1:(first_time-1)) = NaN;
        nlnks(k, 1:(first_time-1)) = NaN;

    end
end







