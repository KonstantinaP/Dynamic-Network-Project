% function test_gtGGPsumrnd

close all
clear all


sigma = .5;
alpha = 100;
tau = 1;

w_rem = 1;
phi = 10;
c_rem = poissrnd(phi*w_rem);


epsilon = 1e-4;
n = 2000;
for i=1:n
    i
    nb(i) = gtGGPsumrnd(alpha, sigma, tau,c_rem, phi); % sample from GIGauss when sigma=0.5
    nb2(i) = gtGGPsumrnd(alpha, sigma+epsilon, tau,c_rem, phi); % Sample from gamma tilted stable
end
figure
ecdf(nb)
hold on
ecdf(nb2)
legend({'Empirical cdf \sigma=1/2', 'Empirical cdf \sigma=1/2+\epsilon'})

% check igauss is correct
% pas = .01;
x0 = linspace(min(nb),max(nb),1000);
stepsize = (max(x0)-min(x0))/length(x0);
out1 = gtGGPsumpdf(x0,alpha, sigma, tau, c_rem, phi);
out2 = gtGGPsumpdf(x0,alpha, sigma+epsilon, tau, c_rem, phi);

figure
plot(x0, out1, 'g')
hold on
plot(x0, out2, 'r')
figure;ecdf(nb);hold on;plot(x0,cumsum(out2)*stepsize,'r');plot(x0,cumsum(out1)*stepsize,'g')
legend({'Empirical cdf \sigma=1/2', 'True cdf \sigma=1/2', 'True cdf \sigma=1/2+\epsilon'})
figure;ecdf(nb2);hold on;plot(x0,cumsum(out2)*stepsize,'r');plot(x0,cumsum(out1)*stepsize,'g')
legend({'Empirical cdf \sigma=1/2+\epsilon', 'True cdf \sigma=1/2', 'True cdf \sigma=1/2+\epsilon'})
sum(out1)*stepsize
sum(out2)*stepsize