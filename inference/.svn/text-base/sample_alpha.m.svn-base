function [alpha, accept]= sample_alpha(weights, counts, alpha, phi, tau, alpha_a, alpha_b, rw_alpha)

accept = 0;
if ~rw_alpha
    
    [alpha] = slice_sample_alpha(weights, counts, alpha, phi, tau, alpha_a, alpha_b);
    
    
else
    [alpha, accept] = sample_alpha_mh(weights, counts, alpha, phi, tau, alpha_a, alpha_b);
    
end

end