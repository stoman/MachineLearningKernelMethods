load('mnist_all.mat')
%training/test data
traina = double(train3(1:1000,:));
trainb = double(train8(1:1000,:));
testa = double(test3);
testb = double(test8);
%regularization parameter
lambda = 0;
%kernel function and its parameters
gamma = 0.00001;
%K = @(x,z) x*z';
K = @(x,z) exp(-gamma.*(bsxfun(@plus, sum(x.^2,2), sum(z.^2,2)') - 2*(x*z')));

%compute x, y, and sample solution 
X = [traina; trainb];
Y = [ones(size(traina,1),1)*-1; ones(size(trainb,1),1)*1];
Xt = [testa; testb];
Yt = [ones(size(testa,1),1)*-1; ones(size(testb,1),1)*1];

%solve the regular dual linear regsression problem with the given kernel
a = (pdist2(X, X, K) + lambda*eye(size(X, 1)))\Y;
%results can be predicted by multiplying with the vector a
predict = @(Xt) (pdist2(Xt, X, K) * a);
predictions = predict(Xt);

fprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1));