%In this file we predict the mnist test data using the kmeans function.
%To do this we try to fit two centroids in the training and test data to
%separate the test data. This approach is quite simple and should be much
%worse than other methods.

%load mnist data
load('mnist_all.mat');

%training/test data
traina = double(train7);
trainb = double(train1);
testa = double(test7);
testb = double(test1);

%create training/test array
testdata = [testa; testb];
trainingdata = [traina; trainb];

%predict values using the kmeans function
groups = kmeans(double([testdata; trainingdata]), 2);
wrong = 0;

%analyze data set A
resulta = groups(1:size(testa,1));
modea = mode(resulta);
wrong = wrong + sum(abs(modea - resulta) > 0);

%analyze data set B
resultb = groups(size(testa,1)+1:size(testa,1)+size(testb,1));
modeb = mode(resultb);
wrong = wrong + sum(abs(modeb - resultb) > 0);

%some assertions assure that the data sets have been indead separated
assert(modea > 0);
assert(modeb > 0);
assert(modea ~= modeb);

%print results
fprintf('works for %d of %d values (%d%%)\n', size(testdata,1)-wrong, size(testdata,1), 100*(size(testdata,1)-wrong)/size(testdata,1));
