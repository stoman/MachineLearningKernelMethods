%In this file we try to find good parameters for the kernel.m file.
%Different values for lambda and gamma are tested for a few training data
%points. We then cross-validate the resulting functions against the
%remaining test data. Executing this file may take some time depending on
%the hardware.
%Author: Stefan Toman (toman@tum.de)

load('mnist_all.mat');

%training/test data
trainingsize = 50;
[X, Y] = testdataset(train3(1:trainingsize,:), train8(1:trainingsize,:));
[Xt, Yt] = testdataset(train3(trainingsize+1:end,:), train8(trainingsize+1:end,:));

%regularization parameter range
lambda = logspace(0, 2, 6);
%gamma range
gamma = linspace(2e-6, 5e-6, 11);

%define a function that should be plotted
map = @(lambda, gamma) predictionquality( ...
    [traina; trainb], ...
    [ones(size(traina,1),1)*-1; ones(size(trainb,1),1)*1], ...
    lambda, ...
    gaussiankernel(gamma), ...
    [testa; testb], ...
    [ones(size(testa,1),1)*-1; ones(size(testb,1),1)*1] ...
)./(size(testa,1)+size(testb,1));

%plot data
[xlambda, xgamma] = meshgrid(lambda, gamma);
quality = arrayfun(map, xlambda, xgamma);
contourf(xlambda, xgamma, quality);
