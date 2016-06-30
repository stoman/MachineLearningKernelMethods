%This function returns a handle to a prediction function using kernels
%given a training set X and Y, a kernel function K and a sample size for
%the Nyström method. This function will not compute the Gram matrix to save
%computation time and memory. The Gram matrix is approximated using the
%Nyström method. The resulting function can predict single data as row
%vectors or several vectors simultaneously given as a matrix.
%Author: Stefan Toman (toman@tum.de)
function predict = funpredictnystrom(X, Y, K, samplesize)
    %compute the decomposition of the Gram amtrix using the Nyström method
    [A, B] = createnystrom(X, K, samplesize);
    
    %solve the regular dual linear regression problem with the given kernel
    a = B\(A\Y);
    %results can be predicted by multiplying with the vector a
    predict = @(Xt) (pdist2(Xt, X, K) * a);
end
