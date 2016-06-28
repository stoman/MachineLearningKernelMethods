function printResults(x, y, w, name, a, b)
    yy = x*w;
    s = sum(abs(sign(y-(a+b)/2)-sign(yy-(a+b)/2)))/2;
    percent = 100*(1-s/size(y,1));
    %as = y == a;
    %sa = sum(abs(sign(y(as)-(a+b)/2)-sign(yy(as)-(a+b)/2)))/2;
    %percenta = 100*(1-sa/size(y(sa),1));
    fprintf('quality of method %s: %f%%\n', name, percent);
end