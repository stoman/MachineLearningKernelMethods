%This function evaluates the quality of predicted results on a test set as
%used in learning.m. x and y are the test data, w are the parameters of the
%prediction function, name is the name of the method in use and a and b are
%the labels of the two classes of objects.
%Author: Stefan Toman (toman@tum.de)
function printresults(x, y, w, name, a, b)
    yy = x*w;
    s = sum(abs(sign(y-(a+b)/2)-sign(yy-(a+b)/2)))/2;
    percent = 100*(1-s/size(y,1));
    fprintf('quality of method %s: %f%%\n', name, percent);
end