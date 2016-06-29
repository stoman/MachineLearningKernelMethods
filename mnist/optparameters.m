load('mnist_all.mat')
%training/test data
traina = double(train3(1:10,:));
trainb = double(train8(1:10,:));
testa = double(test3);
testb = double(test8);
%regularization parameter
lambda = linspace(0.5, 1.5, 5);
%gamma range
gamma = linspace(3e-6, 4e-6, 25);

%compute x, y, and sample solution 
X = [traina; trainb];
Y = [ones(size(traina,1),1)*-1; ones(size(trainb,1),1)*1];
Xt = [testa; testb];
Yt = [ones(size(testa,1),1)*-1; ones(size(testb,1),1)*1];

%
map = @(lambda, gamma) predictionquality( ...
    [traina; trainb], ...
    [ones(size(traina,1),1)*-1; ones(size(trainb,1),1)*1], ...
    lambda, ...
    @(x,z) exp(-gamma.*(bsxfun(@plus, sum(x.^2,2), sum(z.^2,2)') - 2*(x*z'))), ...
    [testa; testb], ...
    [ones(size(testa,1),1)*-1; ones(size(testb,1),1)*1] ...
)./(size(testa,1)+size(testb,1));

[xlambda, xgamma] = meshgrid(lambda, gamma);
quality = arrayfun(map, xlambda, xgamma);
contourf(xlambda, xgamma, quality);