%This file applies kernel methods using the Nyström method to the mnist
%data set. Two digits are separated using different kernels and
%regularization parameters.
%Author: Stefan Toman (toman@tum.de)

%load data
load('../data/mnist_all.mat');

%load functions
addpath('../functions');

%training/test data
[X, Y] = testdataset(train7, train1);
[Xt, Yt] = testdataset(test7, test1);

%sample size
samplesize = 500;

%kernel function and its parameters
%default kernel
%K = defaultkernel();
%Gaussian kernel
h = 500;
gamma = 1;
K = gaussiankernel(gamma/h^2,2,true);

%solve the regular dual linear regression problem with the given kernel
predict = funpredictnystrom(X, Y, K, samplesize);
predictions = predict(Xt);

%print results
fprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1));
