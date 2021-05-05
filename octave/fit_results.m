clear;

pkg load optim;

R_el_1 = 1.0;  % electrode radius

% read data
dat(:,:,1) = dlmread("../results/results_N_ref=3_N_R=4.dat");
dat(:,:,2) = dlmread("../results/results_N_ref=4_N_R=8.dat");

hold on; grid on;
d = 0:0.01:2;

% inspection of data shows superlinear convergence -> estimate error by linear extrapolation
modelfun_point_fit = @(b, x) (b(1) + b(2) * x);
N_datasets = length(dat(1,1,:));
for i=1:length(dat(:,1,1))
  dat_d = zeros(N_datasets,2);
  dat_d(:,1) = 1./2.^[0:1:N_datasets-1];
  dat_d(:,2) = dat(i,2,:);
  b_init = [dat_d(N_datasets,2) 0 2];
  b = polyfit(dat_d(N_datasets-1:N_datasets,1), dat_d(N_datasets-1:N_datasets,2), 1);
  e_fe(i) = abs((b(2) - dat_d(N_datasets,2)) / b(2));
endfor
disp(["Estimated relative error of finite element results = " num2str(max(e_fe))]);

% fit data
modelfun = @(b, x) (1 + b(1) * (x / R_el_1) .^ b(2));
b_init = [0.7331 0.87288];
b = nlinfit (dat(:,1,N_datasets), dat(:,2,N_datasets), modelfun, b_init);
b = [round(b(1)*10000)/10000 round(b(2)*10000)/10000];

% plot data points and fit
for i=N_datasets
 plot(dat(:,1,i), dat(:,2,i), 'kx');  
endfor
d = 0 : 0.01 : 2.0;
plot(d, modelfun(b, d), 'k-');

% plot fit
disp(["Fit f = 1 + " num2str(b(1)) "*(x/R_el_1)^" num2str(b(2))]);

% calculate error of fit
for i=1:length(dat(:,1,N_datasets))
  e_fit(i) = abs((modelfun(b,dat(i,1,N_datasets)) - dat(i,2,N_datasets)) / dat(i,2,N_datasets));
endfor
disp(["Estimated relative error of fit = " num2str(max(e_fit))]);

