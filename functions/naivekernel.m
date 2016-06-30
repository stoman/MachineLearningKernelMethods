%This function returns a function handle for a naive kernel using a
%parameter width. The kernel will create a box above each data point of
%size 1 with length 2*width. The argument nrinputs is optional and
%defaults to 2. If nrinputs is 1 the function will take one argument, if
%nrinputs is 2 it will take two arguments and call the function for one
%argument with the difference of the arguments.
%Author: Stefan Toman (toman@tum.de)
function K = naivekernel(width, nrinputs)
    %optional argument
    if nargin == 1
        nrinputs = 2;
    end
    
    %create function
    if nrinputs == 1
        K = @(x) (sum(x.^2,2)<width)./(2*width);
    elseif nrinputs == 2
        K1 = naivekernel(gamma, 1);
        K = @(x,z) K1(bsxfun(@minus, x, z));
    else
        error('parameter nrinputs should be 1 or 2');
    end
end
