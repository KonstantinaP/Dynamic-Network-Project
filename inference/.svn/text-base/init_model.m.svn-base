function [trZ, weights, K, n, ind] = init_model(Z, ID, alpha, tag, settings)
% INPUT
% Z: cell of length N. Each element is a binry matrix indicating the links
%    between nodes at time t.
% ID: cell of length N. Each element is a vector referring to time t 
%   and contains the unique ids of the nodes at time t.

% NOTE: the max id in ID should be equal to the total number of nodes
% appearing over all the time steps. 

% OUTPUT
% weights: K+1 x N matrix of the sociabilities
% K      : total number of nodes appearing over all time steps.
% ind    : cell structure of length N. It contains the indices of the links
%        i.e. Z_{tij}>0  at each time step.


% NOTE that a node appears at t when it is part of the network interactions at
% time t. That includ the case when a node has 0 links at that time. 

N = length(Z); % N = total # of time steps

for t=1:N
    Zt = Z{t}; % 
    n(t) = size(Zt,1); % n = total # of nodes at each time step
    
end

K = max(cell2mat(ID)); % K = total number of nodes appearing in the dynamic  network of all time steps.

appear =zeros(K+1, N);
for t=1:N
    appear(ID{t}, t) = 1;
    
end
appear(K+1, :) =  alpha;
                        
weights = ones(K+1, N); %Increase the number of nodes by one to account for the unobserved ones.
if tag
    for k=1:K+1
        first_time = find(appear(k, :), 1);
        weights(k, 1:(first_time-1)) = NaN;
    end
end


% Work on the graphs a bit...
if strcmp(settings.typegraph, 'simple')
    issimple = true;
else
    issimple =false;
end

ind=cell(1, N);
nind=cell(1, N);

for t=1:N
    G= Z{t};
    ids = ID{t};
    if issimple % If no self-loops
        G2 = triu((G+G')>0, 1);
        temp = tril(ones(size(G2)));
        
    else
        G2 = triu((G+G')>0);
        temp = tril(ones(size(G2)), -1);        
    end
    [ind1, ind2] = find(G2); % the indices refer to the upper triangle
    
    mapped1 =  ids(ind1);
    mapped2 =  ids(ind2);    
    ind{t} =[mapped1' mapped2'];
    
    
    [nind1, nind2] = find(temp + G2 == 0);
    nind{t} = [ids(nind1)' ids(nind2)'];
    trZ{t} = G2;
end
