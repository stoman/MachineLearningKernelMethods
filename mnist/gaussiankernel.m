%This function returns a function handle for a Gaussian kernel using a
%given parameter gamma. The argument inputs is optional and defaults to 2.
%If it is 1 the function will take one argument, if it is 2 it will take
%two arguments and call the function for one argument with the difference
%of the arguments.
%Author: Stefan Toman (toman@tum.de)
function K = gaussiankernel(gamma, inputs)
    %optional argument
    if nargin == 1
        inputs = 2;
    end
    
    %create function
    if inputs == 1
        K = @(x) exp(-gamma.*sum(x.^2,2));
    elseif inputs == 2
        K1 = gaussiankernel(gamma, 1);
        K = @(x,z) K1(bsxfun(@minus, x, z));
    else
        error('parameter inputs should be 1 or 2');
    end
end
