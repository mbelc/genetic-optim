% The following figures were plotted using the PlotPub library
% https://github.com/masumhabib/PlotPub
% Download and "addtopath" the library before executing the script:
% https://github.com/masumhabib/PlotPub
addpath('C:\Users\Mohamed\Desktop\PlotPub-master\PlotPub-master\lib');


xrange   = [100 80 40];
[~, d]   = size(xrange);


% ----------------------------- CASE 1 -----------------------------
tot_runs = 10;
t1   = zeros(tot_runs,1);
myF1 = zeros(tot_runs,1);
myX1 = zeros(tot_runs,d);
for runs = 1:tot_runs
    tic;
    [myF1(runs),myX1(runs,:)] = myGA(@dejong,xrange);
    t1(runs) = toc;
end
%{
% plot membership functions:
plt = Plot(runs, t1);
% change settings
plt.XLabel = 'Algorithm Run Number';   % xlabel
plt.YLabel = 'Time Complexity'; % ylabel
plt.Legend = ["Time Complexity for different runs"];
%}


% ----------------------------- CASE 2 -----------------------------
tot_runs = 10;
t2   = zeros(tot_runs,1);
myF2 = zeros(tot_runs,1);
myX2 = zeros(tot_runs,d);
for runs = 1:tot_runs
    tic;
    [myF2(runs),myX2(runs,:)] = myGA(@ackley,xrange);
    t2(runs) = toc;
end
%{
% plot membership functions:
plt = Plot(runs, t2);
% change settings
plt.XLabel = 'Algorithm Run Number';   % xlabel
plt.YLabel = 'Time Complexity'; % ylabel
plt.Legend = ["Time Complexity for different runs"];
%}


% ----------------------------- CASE 3 -----------------------------
tot_runs = 10;
t3   = zeros(tot_runs,1);
myF3 = zeros(tot_runs,1);
myX3 = zeros(tot_runs,d);
for runs = 1:tot_runs
    tic;
    [myF3(runs),myX3(runs,:)] = myGA(@rastrigin,xrange);
    t3(runs) = toc;
end
%{
% plot membership functions:
plt = Plot(runs, t3);
% change settings
plt.XLabel = 'Algorithm Run Number';   % xlabel
plt.YLabel = 'Time Complexity'; % ylabel
plt.Legend = ["Time Complexity for different runs"];
%}
