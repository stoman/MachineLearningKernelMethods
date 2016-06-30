%A gradient computation used for the fminunc function in learning.m for the
%gradient descent.
%Author: Stefan Toman (toman@tum.de)
function [L, grad] = gradient(w, x, y)
    L = 1/2*norm(x*w-y);
    if nargout > 1
        grad = x'*(x*w-y);
    end;
end