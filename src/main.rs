use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

fn main() {
    // Set system parameters
    let n = 64;  // number of grid points
    let l = 1.0; // system length
    let dx = l / n as f64;
    
    // Generate input function f(x) = sin(2πx/L)
    let mut f: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let x = i as f64 * dx;
            Complex::new(f64::sin(2.0 * PI * x / l), 0.0)
        })
        .collect();
    
    // Calculate analytical solution for comparison
    let analytical: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * dx;
            -l * l / (4.0 * PI * PI) * f64::sin(2.0 * PI * x / l)
        })
        .collect();

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Perform FFT on f(x)
    fft.process(&mut f);

    // Apply Laplacian operator in Fourier space
    for (i, val) in f.iter_mut().enumerate() {
        let k = if i <= n/2 { i as f64 } else { i as f64 - n as f64 };
        let k = 2.0 * PI * k / l;
        
        // For -k^2 u = f, we solve u = -f/k^2
        if i != 0 {  // For k != 0
            *val = -*val / Complex::new(k * k, 0.0);
        } else {  // For k = 0 (constant term)
            *val = Complex::new(0.0, 0.0);
        }
    }

    // Perform inverse FFT to get the solution
    ifft.process(&mut f);

    // Normalize and print results
    println!("x\tNumerical\tAnalytical\tError");
    for i in 0..n {
        let x = i as f64 * dx;
        let numerical = f[i].re / n as f64;  // FFT normalization
        let error = (numerical - analytical[i]).abs();
        println!("{:.3}\t{:.6}\t{:.6}\t{:.6e}", 
                 x, numerical, analytical[i], error);
    }
}
