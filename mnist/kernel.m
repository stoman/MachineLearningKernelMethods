%This file applies kernel methods to the mnist data set. Two digits are
%separated using different kernels and regularization parameters. Use the
%file optparameters.m to find parameters with good results.
%Author: Stefan Toman (toman@tum.de)

%load data
load('mnist_all.mat');

%training/test data
[X, Y] = testdataset(train3(1:1000,:), train8(1:1000,:));
[Xt, Yt] = testdataset(test3, test8);

%regularization parameter
lambda = 1;
%kernel function and its parameters
%Euclidean kernel
%K = @(x,z) x*z';
%Gaussian kernel
K = gaussiankernel(2.5e-6);

%solve the regular dual linear regsression problem with the given kernel
predict = funpredict(X, Y, lambda, K);
predictions = predict(Xt);

%print results
fprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1));
