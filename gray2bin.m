% Credit: codes developed in this project have been largely inspired by the following references:
% “Gray Code in Matlab – from/to binary and decimal,” matrixlab-examples.com. [Online]. Available: http://www.matrixlab-examples.com/gray-code.html [Accessed Nov. 19, 2019].
% function originally manipulates one scalar in input / output
% function modified to manipulate 2-D CELLS OF STRINGS of scalars
% convert vector of Gray code values to vector of signed binary values


function [B] = gray2bin(G)
[n,m] = size(G);
% initialize B to prealocate space
B = G;
% loop over for each signed binary scalar in vector G
for k=1:n
    for l=1:m
        g = G{k,l};
        % initialize g to prealocate space
        b = g;
        % sign bit is kept
        b(1) = g(1);
        % conversion to signed binary: b(i) = b(i-1) xor g(i)
        for i = 2 : length(g)
            x = xor(str2double(b(i-1)), str2double(g(i)));
            b(i) = num2str(x);
        end
        B{k,l} = b;
    end
end
