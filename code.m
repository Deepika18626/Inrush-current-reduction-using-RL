%% Transformer3P_Qlearning_Full.m
% Full, self-contained MATLAB code:
% - 3-phase simplified transformer energization model (remanent flux surrogate)
% - Tabular Q-learning for choosing closing angle to minimize inrush
% - Policy extraction, inrush heatmap, overlay of policy markers
% - CSV export of results (state, action, peaks, pu)
%
% Save as Transformer3P_Qlearning_Full.m and run in MATLAB.
clear; close all; clc;

%% -----------------------
% 1) Simulation & Transformer parameters
%% -----------------------
S_base = 7.4e6;        % VA (7.4 MVA)
Vp_LL  = 30e3;         % V line-to-line primary
Vs_LL  = 20e3;         % V line-to-line secondary
f      = 50;           % Hz

% Time-stepping
dt     = 5e-5;         % simulation time-step (50 µs)
tmax   = 0.25;         % simulate 250 ms after closing
tvec   = 0:dt:tmax;

% Turns (info only)
Np = 824;
Ns = Np * (Vs_LL/Vp_LL);

% Winding & leakage
R_HV = 1.28;           % ohm
R_LV = 0.26;           % ohm
L_leak_HV = 81.12e-3;  % H
L_leak_LV = 55.58e-3;  % H

% Magnetizing small-signal estimate and nonlinear surrogate
I_nom = S_base/(sqrt(3)*Vp_LL);
Im_assumed = 0.005 * I_nom;    % assumed magnetizing current ~0.5% of rated
Vph = Vp_LL / sqrt(3);
Xm_est = Vph / Im_assumed;
Lm_nom = Xm_est / (2*pi*f);   % base magnetizing inductance (H)

% Nonlinear surrogate parameters (tuneable)
phi_sat = 0.08;      % flux scale (Wb)
alpha_sat = 0.85;    % saturation coefficient

% Per-unit base current
I_base = S_base/(sqrt(3)*Vp_LL);

fprintf('Base current I_base = %.3f A\n', I_base);
fprintf('Estimated Lm_nom = %.3f H\n\n', Lm_nom);

%% -----------------------
% 2) Discretized states/actions for Q-learning
%% -----------------------
openingAngles = 0:30:330;   % states: opening angle (deg)
nOpen = length(openingAngles);

% Remanent flux patterns (Wb) - 5 representative patterns
remanentPatterns = [
    0.0,   0.0,   0.0;
    0.05, -0.02, -0.03;
   -0.03,  0.04, -0.01;
    0.08,  0.08,  0.08;
   -0.06, -0.02,  0.05
];
nRem = size(remanentPatterns,1);

% Combined discrete state count
nStates = nOpen * nRem;

% Actions: closing angles discretized coarsely for speed (every 5°)
closingAngles = 0:5:359;
nActions = length(closingAngles);

%% -----------------------
% 3) Q-learning hyperparameters
%% -----------------------
Q = zeros(nStates, nActions);   % Q-table
alpha = 0.2;       % learning rate
gamma = 0.0;       % discount factor (one-step episodes)
epsilon = 0.25;    % exploration probability
episodes = 5000;   % training episodes

rng(0,'twister');  % reproducible randomness

%% -----------------------
% 4) Training loop (one-step episodes)
%% -----------------------
fprintf('Q-learning: %d episodes starting...\n', episodes);
tic;
for ep = 1:episodes
    % sample random state (opening angle and remanent)
    openIdx = randi(nOpen);
    remIdx = randi(nRem);
    stateIdx = (openIdx-1)*nRem + remIdx;
    alpha_open = openingAngles(openIdx);
    phi_r = remanentPatterns(remIdx,:);

    % epsilon-greedy action selection
    if rand < epsilon
        a = randi(nActions);
    else
        [~, a] = max(Q(stateIdx,:));
    end
    alpha_close = closingAngles(a);

    % simulate energization
    simOut = simulate3phase(alpha_open, alpha_close, phi_r, tvec, ...
                Vp_LL, f, Np, Ns, R_HV, R_LV, L_leak_HV, L_leak_LV, ...
                Lm_nom, phi_sat, alpha_sat, S_base);

    % reward: negative peak per-unit current (minimize inrush)
    Ipeak_pu = max([simOut.Ia_pu, simOut.Ib_pu, simOut.Ic_pu]);
    reward = -Ipeak_pu;

    % update Q
    Q(stateIdx,a) = Q(stateIdx,a) + alpha*(reward - Q(stateIdx,a));
end
trainingTime = toc;
fprintf('Training finished in %.1f s\n\n', trainingTime);

%% -----------------------
% 5) Extract policy & evaluate all states
%% -----------------------
[~, bestActionIdx] = max(Q,[],2);

results_table = []; % prepare for CSV: columns described below

row = 0;
for oi = 1:nOpen
    for ri = 1:nRem
        row = row + 1;
        stateIdx = (oi-1)*nRem + ri;
        alpha_open = openingAngles(oi);
        phi_r = remanentPatterns(ri,:);
        aIdx = bestActionIdx(stateIdx);
        alpha_close = closingAngles(aIdx);

        simOut = simulate3phase(alpha_open, alpha_close, phi_r, tvec, ...
                Vp_LL, f, Np, Ns, R_HV, R_LV, L_leak_HV, L_leak_LV, ...
                Lm_nom, phi_sat, alpha_sat, S_base);

        Ia_peak = simOut.Ia_peak;
        Ib_peak = simOut.Ib_peak;
        Ic_peak = simOut.Ic_peak;
        Ia_pu = simOut.Ia_pu;
        Ib_pu = simOut.Ib_pu;
        Ic_pu = simOut.Ic_pu;
        Ipeak_pu = max([Ia_pu, Ib_pu, Ic_pu]);

        results_table(row,:) = [alpha_open, ri, phi_r(1), phi_r(2), phi_r(3), alpha_close, Ia_peak, Ib_peak, Ic_peak, Ia_pu, Ib_pu, Ic_pu, Ipeak_pu];
    end
end

% Column headers for CSV
headers = {'opening_deg','remIdx','phi_r_a_Wb','phi_r_b_Wb','phi_r_c_Wb','chosen_closing_deg',...
    'Ia_peak_A','Ib_peak_A','Ic_peak_A','Ia_pu','Ib_pu','Ic_pu','Ipeak_pu'};

% Write CSV
outTable = array2table(results_table,'VariableNames',headers);
csvFileName = 'Transformer3P_Qlearning_results.csv';
writetable(outTable,csvFileName);
fprintf('Results exported to %s (rows=%d)\n\n', csvFileName, size(outTable,1));

%% -----------------------
% 6) Visualizations
%   - policy map (opening x rem -> chosen closing)
%   - inrush heatmap for a chosen rem pattern with overlay markers
%   - representative time-series
%% -----------------------

% Policy map
policyMap = zeros(nOpen, nRem);
for oi = 1:nOpen
    for ri = 1:nRem
        idx = (oi-1)*nRem + ri;
        policyMap(oi,ri) = results_table(idx,6); % chosen_closing_deg
    end
end

figure('Name','Policy map (opening × rem) -> chosen closing','NumberTitle','off');
imagesc(policyMap);
colormap(jet); colorbar;
ylabel('Opening angle index'); xlabel('Remanent pattern index');
yticks(1:nOpen); yticklabels(openingAngles);
xticks(1:nRem); xticklabels(compose('R%d',1:nRem));
title('Policy map: value = chosen closing angle (deg)');

% Inrush heatmap for a chosen rem pattern index (choose remIdx = 2 by default)
remIdx_eval = 2;
phi_r_eval = remanentPatterns(remIdx_eval,:);
Iheatmap = zeros(nOpen, nActions);

fprintf('Computing inrush heatmap for remanent pattern index %d (this may take a moment)...\n', remIdx_eval);
for oi = 1:nOpen
    alpha_open = openingAngles(oi);
    for ai = 1:nActions
        alpha_close = closingAngles(ai);
        simOut = simulate3phase(alpha_open, alpha_close, phi_r_eval, tvec, ...
                Vp_LL, f, Np, Ns, R_HV, R_LV, L_leak_HV, L_leak_LV, ...
                Lm_nom, phi_sat, alpha_sat, S_base);
        Iheatmap(oi,ai) = max([simOut.Ia_pu, simOut.Ib_pu, simOut.Ic_pu]);
    end
end

figure('Name','Inrush heatmap (opening × closing)','NumberTitle','off');
imagesc(closingAngles, openingAngles, Iheatmap);
set(gca,'YDir','normal');
xlabel('Closing angle (deg)'); ylabel('Opening angle (deg)');
colorbar; title(sprintf('Peak inrush (p.u.) for remanent pattern R%d', remIdx_eval));

% Overlay agent policy for this rem pattern
% For each opening angle, find the chosen closing angle and plot marker
policyAngles_for_rem = zeros(1,nOpen);
for oi = 1:nOpen
    stateIdx = (oi-1)*nRem + remIdx_eval;
    bestIdx = bestActionIdx(stateIdx);
    policyAngles_for_rem(oi) = closingAngles(bestIdx);
end

hold on;
plot(policyAngles_for_rem, openingAngles, 'ks','MarkerFaceColor','w','MarkerSize',8);
legend('Agent chosen closing','Location','southoutside');

% Representative time-series (pick middle state)
rep_index = ceil(size(results_table,1)/2);
rep_alpha_open = results_table(rep_index,1);
rep_remIdx = results_table(rep_index,2);
rep_alpha_close = results_table(rep_index,6);
rep_phi = [results_table(rep_index,3), results_table(rep_index,4), results_table(rep_index,5)];

repSim = simulate3phase(rep_alpha_open, rep_alpha_close, rep_phi, tvec, ...
    Vp_LL, f, Np, Ns, R_HV, R_LV, L_leak_HV, L_leak_LV, ...
    Lm_nom, phi_sat, alpha_sat, S_base);

figure('Name','Representative time-series','NumberTitle','off','Position',[100 100 1000 700]);
subplot(3,1,1);
plot(tvec, repSim.va); hold on; plot(tvec, repSim.vb); plot(tvec, repSim.vc);
legend('va','vb','vc'); ylabel('Voltage (V)'); title(sprintf('Phase voltages (open=%d, close=%d, rem=%d)', rep_alpha_open, rep_alpha_close, rep_remIdx));
grid on;
subplot(3,1,2);
plot(tvec, repSim.ia); hold on; plot(tvec, repSim.ib); plot(tvec, repSim.ic);
legend('ia','ib','ic'); ylabel('Current (A)'); title('Phase currents (magnetizing branch)');
grid on;
subplot(3,1,3);
plot(tvec, repSim.phi_a); hold on; plot(tvec, repSim.phi_b); plot(tvec, repSim.phi_c);
legend('\phi_a','\phi_b','\phi_c'); ylabel('Flux (Wb)'); xlabel('Time (s)'); title('Magnetizing fluxes');
grid on;

%% -----------------------
% 7) Print a short summary table to console (first 12 rows)
%% -----------------------
disp('First 12 results (opening_deg, remIdx, chosen_close_deg, Ipeak_pu):');
for i=1:min(12,size(results_table,1))
    fprintf('Open %3d°, Rem %d -> Close %3d°   Ipeak_pu = %.3f\n', ...
        results_table(i,1), results_table(i,2), results_table(i,6), results_table(i,13));
end

fprintf('\nCSV file "%s" contains full results (columns: %s)\n', csvFileName, strjoin(headers,', '));
fprintf('All done.\n');

%% ============================
% Function: simulate3phase
% All needed params passed explicitly to avoid scope issues.
% Returns struct OUT with fields:
% OUT.t, OUT.va,vb,vc, OUT.ia,ib,ic, OUT.phi_a,phi_b,phi_c,
% OUT.Ia_peak,Ib_peak,Ic_peak, OUT.Ia_pu,Ib_pu,Ic_pu
%% ============================
function OUT = simulate3phase(alpha_open_deg, alpha_close_deg, phi_r_init, t, ...
    Vp_LL, f, Np, Ns, R_HV, R_LV, L_leak_HV, L_leak_LV, ...
    Lm_nom, phi_sat, alpha_sat, S_base)

    dt = t(2) - t(1);
    w = 2*pi*f;
    shifts = [0, -2*pi/3, 2*pi/3];
    Vph_amp = sqrt(2) * (Vp_LL / sqrt(3)); % phase voltage peak

    n = numel(t);
    va = zeros(1,n); vb = zeros(1,n); vc = zeros(1,n);
    ia = zeros(1,n); ib = zeros(1,n); ic = zeros(1,n);
    phi_a = zeros(1,n); phi_b = zeros(1,n); phi_c = zeros(1,n);

    % initial remanent fluxes
    phi_a(1) = phi_r_init(1);
    phi_b(1) = phi_r_init(2);
    phi_c(1) = phi_r_init(3);

    % offset source phase by closing angle to simulate closing instant
    phaseOffset = deg2rad(alpha_close_deg);

    for k = 2:n
        tt = t(k);
        % source voltages
        va(k) = Vph_amp * sin(w*tt + phaseOffset + shifts(1));
        vb(k) = Vph_amp * sin(w*tt + phaseOffset + shifts(2));
        vc(k) = Vph_amp * sin(w*tt + phaseOffset + shifts(3));

        % estimate di/dt safely (backward difference)
        if k == 2
            dia_dt_a = 0; dia_dt_b = 0; dia_dt_c = 0;
        else
            dia_dt_a = (ia(k-1) - ia(k-2)) / dt;
            dia_dt_b = (ib(k-1) - ib(k-2)) / dt;
            dia_dt_c = (ic(k-1) - ic(k-2)) / dt;
        end

        % leakage voltage drop approx: R*i + L*di/dt
        vdrop_a = R_HV * ia(k-1) + L_leak_HV * dia_dt_a;
        vdrop_b = R_HV * ib(k-1) + L_leak_HV * dia_dt_b;
        vdrop_c = R_HV * ic(k-1) + L_leak_HV * dia_dt_c;

        % voltage across magnetizing branch
        vcore_a = va(k) - vdrop_a;
        vcore_b = vb(k) - vdrop_b;
        vcore_c = vc(k) - vdrop_c;

        % integrate flux-linkage phi: v = d(phi)/dt
        phi_a(k) = phi_a(k-1) + vcore_a * dt;
        phi_b(k) = phi_b(k-1) + vcore_b * dt;
        phi_c(k) = phi_c(k-1) + vcore_c * dt;

        % nonlinear magnetizing inductance surrogate
        Lm_a = Lm_nom * (1 - alpha_sat * tanh(phi_a(k)/phi_sat));
        Lm_b = Lm_nom * (1 - alpha_sat * tanh(phi_b(k)/phi_sat));
        Lm_c = Lm_nom * (1 - alpha_sat * tanh(phi_c(k)/phi_sat));

        % magnetizing current (phi = Lm * im)
        ia(k) = phi_a(k) / Lm_a;
        ib(k) = phi_b(k) / Lm_b;
        ic(k) = phi_c(k) / Lm_c;

        % safety clamp (avoid numerical blow-ups)
        clamp = 1e7;
        ia(k) = max(-clamp, min(clamp, ia(k)));
        ib(k) = max(-clamp, min(clamp, ib(k)));
        ic(k) = max(-clamp, min(clamp, ic(k)));
    end

    % peaks and per-unit
    I_base = S_base / (sqrt(3) * Vp_LL);
    Ia_peak = max(abs(ia)); Ib_peak = max(abs(ib)); Ic_peak = max(abs(ic));
    OUT.t = t;
    OUT.va = va; OUT.vb = vb; OUT.vc = vc;
    OUT.ia = ia; OUT.ib = ib; OUT.ic = ic;
    OUT.phi_a = phi_a; OUT.phi_b = phi_b; OUT.phi_c = phi_c;
    OUT.Ia_peak = Ia_peak; OUT.Ib_peak = Ib_peak; OUT.Ic_peak = Ic_peak;
    OUT.Ia_pu = Ia_peak / I_base; OUT.Ib_pu = Ib_peak / I_base; OUT.Ic_pu = Ic_peak / I_base;
end