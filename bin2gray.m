% Credit: codes developed in this project have been largely inspired by the following references:
% “Gray Code in Matlab – from/to binary and decimal,” matrixlab-examples.com. [Online]. Available: http://www.matrixlab-examples.com/gray-code.html [Accessed Nov. 19, 2019].
% function originally manipulates one scalar in input / output
% function modified to manipulate 2-D CELLS OF STRINGS of scalars
% convert vector of signed binary values to vector of Gray code values


function [G] = bin2gray(B)
[n,m] = size(B);
% initialize G to prealocate space
G = B;
% loop over for each signed binary scalar in vector B
for k=1:n
    for l=1:m
        b = B{k,l};
        % initialize g to prealocate space
        g = b;
        % sign bit is kept
        g(1) = b(1);
        % conversion to Gray code: g(i) = b(i-1) xor b(i)
        for i = 2 : length(b)
            x = xor(str2double(b(i-1)), str2double(b(i)));
            g(i) = num2str(x);
        end
        G{k,l} = g;
    end
end
