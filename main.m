function x = main(y, params)
    %MAIN Edge-preserving image smoothing.
    % y: real double H-by-W or H-by-W-by-C image, H,W >= 3.
    % params: lambda1, lambda2, rho1, rho2, p1, p2, max_iter.
    % Requires Image Processing Toolbox (psf2otf).
    lam1 = params.lambda1; lam2 = params.lambda2;
    rho1 = params.rho1; rho2 = params.rho2;
    p1 = params.p1; p2 = params.p2;

    [H, W, C] = size(y);

    % Periodic derivative operators.
    dx = [1, -1]; dy = [1; -1];
    F_dx = psf2otf(dx, [H, W]); F_dy = psf2otf(dy, [H, W]);

    dxx = [1, -2, 1]; dyy = [1; -2; 1];
    dxy = 0.25 * [-1, 0, 1; 0, 0, 0; 1, 0, -1];
    F_dxx = psf2otf(dxx, [H, W]);
    F_dyy = psf2otf(dyy, [H, W]);
    F_dxy = psf2otf(dxy, [H, W]);

    Denom = 1 + rho1 * (abs(F_dx).^2 + abs(F_dy).^2) + ...
                rho2 * (abs(F_dxx).^2 + abs(F_dyy).^2 + 2 * abs(F_dxy).^2);

    x = y;
    w_dx = zeros(H, W, C); w_dy = zeros(H, W, C);
    u1_dx = zeros(H, W, C); u1_dy = zeros(H, W, C);

    z_xx = zeros(H, W, C); z_yy = zeros(H, W, C); z_xy = zeros(H, W, C);
    u2_xx = zeros(H, W, C); u2_yy = zeros(H, W, C); u2_xy = zeros(H, W, C);

    for k = 1:params.max_iter

        % Image update.
        rhs_grad = conj(F_dx) .* fft2(w_dx - u1_dx) + ...
                   conj(F_dy) .* fft2(w_dy - u1_dy);
        rhs_hess = conj(F_dxx) .* fft2(z_xx - u2_xx) + ...
                   conj(F_dyy) .* fft2(z_yy - u2_yy) + ...
                   2 * conj(F_dxy) .* fft2(z_xy - u2_xy);

        rhs_total = fft2(y) + rho1 * rhs_grad + rho2 * rhs_hess;
        x = real(ifft2(rhs_total ./ Denom));

        fx = fft2(x);
        Dx = real(ifft2(fx .* F_dx)); Dy = real(ifft2(fx .* F_dy));
        Dxx = real(ifft2(fx .* F_dxx)); Dyy = real(ifft2(fx .* F_dyy)); Dxy = real(ifft2(fx .* F_dxy));

        % Joint-channel gradient shrinkage.
        T_dx = Dx + u1_dx; T_dy = Dy + u1_dy;
        mag_sq = sum(T_dx.^2 + T_dy.^2, 3);
        M = sqrt(mag_sq);
        M = max(M, 1e-9);

        threshold = (lam1 / rho1) .* (M .^ (p1 - 1.0));
        scale_factor = max(0, 1 - threshold ./ M);
        scale_factor = repmat(scale_factor, [1, 1, C]);

        w_dx = scale_factor .* T_dx;
        w_dy = scale_factor .* T_dy;

        % Per-channel Hessian spectral shrinkage.
        for c = 1:C
            K_xx = Dxx(:,:,c) + u2_xx(:,:,c);
            K_yy = Dyy(:,:,c) + u2_yy(:,:,c);
            K_xy = Dxy(:,:,c) + u2_xy(:,:,c);

            tr = K_xx + K_yy;
            diff = K_xx - K_yy;
            delta = sqrt(diff.^2 + 4 * K_xy.^2);

            eig1 = (tr + delta) / 2;
            eig2 = (tr - delta) / 2;

            eig1_new = generalized_shrinkage(eig1, lam2, rho2, p2);
            eig2_new = generalized_shrinkage(eig2, lam2, rho2, p2);

            is_singular = delta < 1e-9;
            c1 = zeros(size(delta));

            idx = ~is_singular;
            c1(idx) = (eig1_new(idx) - eig2_new(idx)) ./ delta(idx);

            idx_s = is_singular;
            if any(idx_s(:))
                safe_denom = eig1(idx_s);
                safe_denom(abs(safe_denom) < 1e-9) = 1e-9;
                c1(idx_s) = eig1_new(idx_s) ./ safe_denom;
            end

            c2 = eig1_new - c1 .* eig1;

            z_xx(:,:,c) = c1 .* K_xx + c2;
            z_yy(:,:,c) = c1 .* K_yy + c2;
            z_xy(:,:,c) = c1 .* K_xy;
        end

        % Dual updates.
        u1_dx = u1_dx + (Dx - w_dx); u1_dy = u1_dy + (Dy - w_dy);
        u2_xx = u2_xx + (Dxx - z_xx); u2_yy = u2_yy + (Dyy - z_yy); u2_xy = u2_xy + (Dxy - z_xy);
    end
end

function val_new = generalized_shrinkage(val, lambda, rho, p)
    sign_val = sign(val);
    abs_val = abs(val);
    tau = (lambda / rho);
    abs_val_safe = max(abs_val, 1e-9);

    shrink = tau .* (abs_val_safe .^ (p - 1.0));
    mag_new = max(0, abs_val - shrink);

    val_new = sign_val .* mag_new;
end
