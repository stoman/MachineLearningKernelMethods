%In this file we try to find good parameters for the kernel.m file.
%Different values for lambda and gamma are tested for a few training data
%points. We then cross-validate the resulting functions against the
%remaining test data. Executing this file may take some time depending on
%the hardware.
%Author: Stefan Toman (toman@tum.de)

load('mnist_all.mat');

%training/test data
trainingsize = 50;
traina = double(train3(1:trainingsize,:));
trainb = double(train8(1:trainingsize,:));
testa = double(train3(trainingsize+1:end,:));
testb = double(train8(trainingsize+1:end,:));

%regularization parameter range
lambda = logspace(0, 2, 6);
%gamma range
gamma = linspace(2e-6, 5e-6, 11);

%compute x, y, and sample solution 
X = [traina; trainb];
Y = [ones(size(traina,1),1)*-1; ones(size(trainb,1),1)*1];
Xt = [testa; testb];
Yt = [ones(size(testa,1),1)*-1; ones(size(testb,1),1)*1];

%define a function that should be plotted
map = @(lambda, gamma) predictionquality( ...
    [traina; trainb], ...
    [ones(size(traina,1),1)*-1; ones(size(trainb,1),1)*1], ...
    lambda, ...
    @(x,z) exp(-gamma.*(bsxfun(@plus, sum(x.^2,2), sum(z.^2,2)') - 2*(x*z'))), ...
    [testa; testb], ...
    [ones(size(testa,1),1)*-1; ones(size(testb,1),1)*1] ...
)./(size(testa,1)+size(testb,1));

%plot data
[xlambda, xgamma] = meshgrid(lambda, gamma);
quality = arrayfun(map, xlambda, xgamma);
contourf(xlambda, xgamma, quality);
