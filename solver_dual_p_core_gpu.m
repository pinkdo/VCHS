function x_out = solver_dual_p_core_gpu(y, params)
%SOLVER_DUAL_P_CORE_GPU GPU backend for main.
%   The update order and parameters match main. The input is
%   converted once to a gpuArray and the result is gathered once at return.
%   Double precision is retained to match the CPU implementation.
%
%   Required fields in params:
%     lambda1, lambda2, rho1, rho2, p1, p2, max_iter
%   Set params.use_gpu = true in the caller. A supported GPU and the
%   Parallel Computing Toolbox are required.

    if ~isa(y, 'gpuArray')
        y = gpuArray(y);
    end
    % Convert on the device so the GPU backend matches the CPU double path.
    y = double(y);

    [H, W, C] = size(y);

    % Periodic finite-difference symbols. psf2otf is evaluated on the CPU
    % once; all subsequent arrays remain on the GPU.
    dx = [1, -1];
    dy = [1; -1];
    dxx = [1, -2, 1];
    dyy = [1; -2; 1];
    dxy = 0.25 * [-1, 0, 1; 0, 0, 0; 1, 0, -1];

    F_dx  = gpuArray(psf2otf(dx,  [H, W]));
    F_dy  = gpuArray(psf2otf(dy,  [H, W]));
    F_dxx = gpuArray(psf2otf(dxx, [H, W]));
    F_dyy = gpuArray(psf2otf(dyy, [H, W]));
    F_dxy = gpuArray(psf2otf(dxy, [H, W]));

    rho1 = params.rho1;
    rho2 = params.rho2;
    lam1 = params.lambda1;
    lam2 = params.lambda2;
    p1 = params.p1;
    p2 = params.p2;

    Denom = 1 + rho1 * (abs(F_dx).^2 + abs(F_dy).^2) + ...
                rho2 * (abs(F_dxx).^2 + abs(F_dyy).^2 + ...
                        2 * abs(F_dxy).^2);

    % All primal, auxiliary, and dual variables stay on the GPU.
    x = y;
    w_dx = zeros(H, W, C, 'like', y);
    w_dy = zeros(H, W, C, 'like', y);
    u1_dx = zeros(H, W, C, 'like', y);
    u1_dy = zeros(H, W, C, 'like', y);
    z_xx = zeros(H, W, C, 'like', y);
    z_yy = zeros(H, W, C, 'like', y);
    z_xy = zeros(H, W, C, 'like', y);
    u2_xx = zeros(H, W, C, 'like', y);
    u2_yy = zeros(H, W, C, 'like', y);
    u2_xy = zeros(H, W, C, 'like', y);

    for k = 1:params.max_iter %#ok<NASGU>
        % x update
        rhs_grad = conj(F_dx) .* fft2(w_dx - u1_dx) + ...
                   conj(F_dy) .* fft2(w_dy - u1_dy);
        rhs_hess = conj(F_dxx) .* fft2(z_xx - u2_xx) + ...
                   conj(F_dyy) .* fft2(z_yy - u2_yy) + ...
                   2 * conj(F_dxy) .* fft2(z_xy - u2_xy);
        rhs_total = fft2(y) + rho1 * rhs_grad + rho2 * rhs_hess;
        x = real(ifft2(rhs_total ./ Denom));

        % Derivatives
        fx = fft2(x);
        Dx  = real(ifft2(fx .* F_dx));
        Dy  = real(ifft2(fx .* F_dy));
        Dxx = real(ifft2(fx .* F_dxx));
        Dyy = real(ifft2(fx .* F_dyy));
        Dxy = real(ifft2(fx .* F_dxy));

        % Joint RGB gradient update
        T_dx = Dx + u1_dx;
        T_dy = Dy + u1_dy;
        M = sqrt(sum(T_dx.^2 + T_dy.^2, 3));
        M = max(M, 1e-9);
        threshold = (lam1 / rho1) .* (M .^ (p1 - 1.0));
        scale_factor = max(0, 1 - threshold ./ M);
        scale_factor = repmat(scale_factor, [1, 1, C]);
        w_dx = scale_factor .* T_dx;
        w_dy = scale_factor .* T_dy;

        % Per-channel symmetric Hessian update. The masks and branch are
        % evaluated on the GPU without gathering a full image.
        for c = 1:C
            K_xx = Dxx(:,:,c) + u2_xx(:,:,c);
            K_yy = Dyy(:,:,c) + u2_yy(:,:,c);
            K_xy = Dxy(:,:,c) + u2_xy(:,:,c);

            tr = K_xx + K_yy;
            diff = K_xx - K_yy;
            delta = sqrt(diff.^2 + 4 * K_xy.^2);
            eig1 = (tr + delta) / 2;
            eig2 = (tr - delta) / 2;

            eig1_new = solver_dual_p_shrink_gpu(eig1, lam2, rho2, p2);
            eig2_new = solver_dual_p_shrink_gpu(eig2, lam2, rho2, p2);

            singular = delta < 1e-9;
            c1_regular = (eig1_new - eig2_new) ./ max(delta, 1e-9);
            safe_eig1 = eig1;
            safe_eig1(abs(safe_eig1) < 1e-9) = 1e-9;
            c1_singular = eig1_new ./ safe_eig1;
            c1 = c1_regular;
            c1(singular) = c1_singular(singular);

            c2 = eig1_new - c1 .* eig1;
            z_xx(:,:,c) = c1 .* K_xx + c2;
            z_yy(:,:,c) = c1 .* K_yy + c2;
            z_xy(:,:,c) = c1 .* K_xy;
        end

        % Multiplier update
        u1_dx = u1_dx + (Dx - w_dx);
        u1_dy = u1_dy + (Dy - w_dy);
        u2_xx = u2_xx + (Dxx - z_xx);
        u2_yy = u2_yy + (Dyy - z_yy);
        u2_xy = u2_xy + (Dxy - z_xy);
    end

    % This gather also synchronizes the GPU before the caller stops timing.
    x_out = gather(x);
end

function val_new = solver_dual_p_shrink_gpu(val, lambda, rho, p)
    sign_val = sign(val);
    abs_val = abs(val);
    abs_val_safe = max(abs_val, 1e-9);
    shrink = (lambda / rho) .* (abs_val_safe .^ (p - 1.0));
    val_new = sign_val .* max(0, abs_val - shrink);
end
