%This function returns a function handle for a default kernel (the inner
%product).
%Author: Stefan Toman (toman@tum.de)
function K = defaultkernel()
    K = @(x,z) x*z';
end
