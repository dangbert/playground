% based on:
%  http://www.math.utah.edu/lab/ms/matlab/matlab.html

% note: when a line ends with a semicolon the output is supressed
fprintf("----------matrices:----------\n")
a = [1 2; 2 1]

fprintf("multiplying a*a\n")
a * a

b = [1 2; 0 1]

fprintf("multiplying b*a\n")
b*a

fprintf("s = b+a\n")
s = b+a

fprintf("det(s) = %.2f, inv(s)=\n", det(s))
inv(s)

fprintf("a * inv(s) is the same as a/s (note direction of slash)\n")
a * inv(s)
a/s

fprintf("inv(a) * s is the same as a\\s (note direction of slash)\n")
inv(a) * s
a\s

fprintf("use zeros(3,3) to create a matrix of zeros\n.")
fprintf("Use eye(n) to create an identity matrix.  eye(3)=\n");
eye(3)

fprintf("matrix indexing:\n")
A = [1 2 3; 2 3 4; 3 4 5; 4 5 6]
fprintf("A(:, 2:3) =\n")
A(:, 2:3)
fprintf("update entries of A meeting condition\n")
A(A == 3) = 333
 


fprintf("\n\n----------vectors:----------\n")
fprintf("solving the matrix equation Ax = b\n")
b = [1; 0]
fprintf("x = a\\b = inv(a)*b\n")
x = a\b
fprintf("checking solution: a*x=\n")
a * x

fprintf("\n\n----------loops:----------\n")
a = [0.8 0.1; 0.2 0.9]
x = [1;0]

% imagine the values in x reprensent the % of an island's population on the west
%   and east half of the island respectively.
% The state of the population one unit of time later is given by the rule y = ax.
%   (someone on the west thus moves east with a probability of 0.2).

verbose = false;
% i will take on every value in the (inclusive) range [1,20]
% we are simulating the islands population movement over 20 units of time...
for i = 1:20
  x = a*x;
  if verbose
    fprintf("at i=%i, x=\n", i)
  end
end

fprintf("final value of x:\n")
x


fprintf("\n\n----------graphing:----------\n")
fprintf("a function of 1 var\n")
% t takes on values in the range [0:4.8] (getting as close to 5 as possible)
t = 0:.3:5
y = sin(t);
plot(t, y);
fprintf('(PAUSED)');
pause;

fprintf("a function of 2 vars\n")
% create a matrix with entries as points of a grid in the square -2 <= x <= 2, -2 <= y <= 2
%   each grid square has dimensions 0.2 units x 0.2 units
[x,y] = meshgrid(-2:.2:2, -2:.2:2);
% matrix of the values of the function at each grid point:
z = x .* exp(-x.^2 - y.^2);
% construct the graph:
surf(x,y,z);
fprintf('(PAUSED)');
pause;


%function[] = myPause()
%  fprintf('(PAUSED)');
%  pause;
