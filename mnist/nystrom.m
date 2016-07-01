%This file applies kernel methods using the Nyström method to the mnist
%data set. Two digits are separated using different kernels and
%regularization parameters.
%Author: Stefan Toman (toman@tum.de)

%load data
load('../data/mnist_all.mat');

%load functions
addpath('../functions');

%training/test data
[X, Y] = testdataset(train3(1:2000,:), train8(1:2000,:));
[Xt, Yt] = testdataset(test3, test8);

%sample size
samplesize = 500;

%kernel function and its parameters
%default kernel
%K = defaultkernel();
%Gaussian kernel
h = 200;
gamma = 1;
K = gaussiankernel(gamma/2/h^2);

%solve the regular dual linear regsression problem with the given kernel
predict = funpredictnystrom(X, Y, K, samplesize);
predictions = predict(Xt);

%print results
fprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1));
