* Find elliptical covers using MINLP
$if not set n $set n 30
$if not set p $set p 4

set i /1*%n%/;
set j /1*%p%/;
alias(j,j1,j2);
parameter xi(i), yi(i);
xi(i) = uniform(0,1);
yi(i) = uniform(0,1);

variable xj(j), yj(j), objval;
binary variable z(i,j1,j2);
equations assign(i), c_dist(i,j1,j2);
assign(i)..
    sum((j1,j2)$(ord(j1) < ord(j2)), z(i,j1,j2)) =e= 2;
c_dist(i,j1,j2)$(ord(j1) < ord(j2))..
    objval =g= z(i,j1,j2)*(sqrt(sqr(xi(i)-xj(j1)) + sqr(yi(i)-yj(j1))) + sqrt(sqr(xi(i)-xj(j2)) + sqr(yi(i)-yj(j2))));
model find_ellipse /c_dist, assign/;
option minlp=baron, optcr=0;
solve find_ellipse min objval using minlp;
