function predict = funpredict(X, Y, lambda, K)
    %solve the regular dual linear regsression problem with the given kernel
    a = (pdist2(X, X, K) + lambda*eye(size(X, 1)))\Y;
    %results can be predicted by multiplying with the vector a
    predict = @(Xt) (pdist2(Xt, X, K) * a);
end