%This function returns a handle to a prediction function using kernels
%given a training set X and Y, a regularization parameter lambda and a
%kernel function K. The resulting function can predict single data as row
%vectors or several vectors simultaneously given as a matrix.
%Author: Stefan Toman (toman@tum.de)
function predict = funpredict(X, Y, lambda, K)
    %solve the regular dual linear regsression problem with the given kernel
    a = (pdist2(X, X, K) + lambda*eye(size(X, 1)))\Y;
    %results can be predicted by multiplying with the vector a
    predict = @(Xt) (pdist2(Xt, X, K) * a);
end