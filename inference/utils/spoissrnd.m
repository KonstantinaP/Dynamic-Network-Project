%% Sampler truncated Poisson
function x = spoissrnd(lambda, sigma)

% Sample from a zero-truncated Poisson distribution
x = ceil(sigma).*ones(size(lambda));
ind = (lambda > 1e-5); % below this value, x=ceil(sigma) w. very high proba
lambda_ind = lambda(ind);    
x(ind) = poissinv(exp(-lambda_ind) +rand(size(lambda_ind)).*(1 - exp(-lambda_ind)), lambda_ind);
end