%This function returns a function handle for a Euclidean kernel. The
%argument nrinputs is optional and defaults to 2. If nrinputs is 1 the
%function will take one argument, if nrinputs is 2 it will take two
%arguments and call the function for one argument with the difference of
%the arguments.
%Author: Stefan Toman (toman@tum.de)
function K = euclideankernel(nrinputs)
    %optional argument
    if nargin == 0
        nrinputs = 2;
    end
    
    %create function
    if nrinputs == 1
        K = @(x) sum(x*x',2);
    elseif nrinputs == 2
        K1 = euclideankernel(1);
        K = @(x,z) K1(bsxfun(@minus, x, z));
    else
        error('parameter nrinputs should be 1 or 2');
    end
end
