%This function creates a data set given two types of objects. X is the
%concatenation of the rows of both sets, Y is a vector containing 1 for all
%objects in the first set and -1 for all objects in the second set. The
%test data are also converted to doubles.
%Author: Stefan Toman (toman@tum.de)
function [X, Y] = testdataset(seta, setb)
    X = [double(seta); double(setb)];
    Y = [ones(size(seta,1),1)*-1; ones(size(setb,1),1)*1];
end