use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, Axis, s};

use statrs::statistics::Statistics;
use pyo3::types::PyDict;
use numpy::ToPyArray;

/// Calculate technical indicators using Rust for high performance
#[pymodule]
fn tick_analysis_processing(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_technical_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_correlation, m)?)?;
    Ok(())
}

/// Calculate technical indicators for price data
#[pyfunction]
fn calculate_technical_indicators(
    prices: &PyArray1<f64>,
    volumes: &PyArray1<f64>,
    window_size: usize,
) -> PyResult<PyObject> {
    let prices = unsafe { prices.as_array() };
    let volumes = unsafe { volumes.as_array() };
    
    // Convert to Rust arrays for processing
    let prices_arr = Array1::from_vec(prices.to_vec());
    let volumes_arr = Array1::from_vec(volumes.to_vec());
    
    // Calculate indicators in parallel (fix join usage)
    let (sma, ema) = rayon::join(
        || calculate_moving_averages(&prices_arr, window_size),
        || calculate_ema(&prices_arr, window_size),
    );
    let (rsi, macd) = rayon::join(
        || calculate_rsi(&prices_arr, window_size),
        || calculate_macd(&prices_arr),
    );
    let bb = calculate_bollinger_bands(&prices_arr, window_size);
    
    // Calculate VWAP
    let vwap = calculate_vwap(&prices_arr, &volumes_arr, window_size);
    
    // Return results as Python dictionary
    Python::with_gil(|py| {
        let result = PyDict::new(py);
        result.set_item("sma", sma.to_pyarray(py))?;
        result.set_item("ema", ema.to_pyarray(py))?;
        result.set_item("rsi", rsi.to_pyarray(py))?;
        result.set_item("macd", macd.to_pyarray(py))?;
        result.set_item("bollinger_bands", bb.to_pyarray(py))?;
        result.set_item("vwap", vwap.to_pyarray(py))?;
        Ok(result.into())
    })
}

/// Calculate basic statistics for price data
#[pyfunction]
fn calculate_statistics(
    data: &PyArray2<f64>,
    axis: Option<usize>,
) -> PyResult<PyObject> {
    let data = unsafe { data.as_array() };
    let axis = axis.unwrap_or(0);
    
    // Calculate statistics in parallel
    let (mean, std) = rayon::join(
        || data.mean_axis(Axis(axis)).expect("mean_axis returned None").to_owned(),
        || data.std_axis(Axis(axis), 0.0).to_owned(),
    );
    // min and max sequentially (not parallel)
    let min = data.map_axis(Axis(axis), |row| row.min()).to_owned();
    let max = data.map_axis(Axis(axis), |row| row.max()).to_owned();
    
    // Return results as Python dictionary
    Python::with_gil(|py| {
        let result = PyDict::new(py);
        result.set_item("mean", mean.to_pyarray(py))?;
        result.set_item("std", std.to_pyarray(py))?;
        result.set_item("min", min.to_pyarray(py))?;
        result.set_item("max", max.to_pyarray(py))?;
        Ok(result.into())
    })
}

/// Calculate correlation matrix for price data
#[pyfunction]
fn calculate_correlation(data: &PyArray2<f64>) -> PyResult<PyObject> {
    let data = unsafe { data.as_array() };
    let p = data.ncols();
    
    // Calculate correlation matrix in parallel
    let correlation = Array2::from_shape_fn((p, p), |(i, j)| {
        let x = data.slice(s![.., i]).to_owned();
        let y = data.slice(s![.., j]).to_owned();
        calculate_pearson_correlation(&x, &y)
    });
    
    // Return correlation matrix
    Python::with_gil(|py| {
        Ok(correlation.to_pyarray(py).into())
    })
}

// Helper functions

fn calculate_moving_averages(prices: &Array1<f64>, window_size: usize) -> Array1<f64> {
    prices.windows(window_size)
        .into_iter()
        .map(|window| window.mean())
        .collect()
}

fn calculate_ema(prices: &Array1<f64>, window_size: usize) -> Array1<f64> {
    let alpha = 2.0 / (window_size as f64 + 1.0);
    let mut ema = Array1::zeros(prices.len());
    ema[0] = prices[0];
    
    for i in 1..prices.len() {
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i-1];
    }
    
    ema
}

fn calculate_rsi(prices: &Array1<f64>, window_size: usize) -> Array1<f64> {
    let deltas = &prices.slice(s![1..]) - &prices.slice(s![..-1]);
    let gains = deltas.mapv(|x| if x > 0.0 { x } else { 0.0 });
    let losses = deltas.mapv(|x| if x < 0.0 { -x } else { 0.0 });
    let avg_gain: Array1<f64> = gains.windows(window_size)
        .into_iter()
        .map(|window| window.mean())
        .collect();
    let avg_loss: Array1<f64> = losses.windows(window_size)
        .into_iter()
        .map(|window| window.mean())
        .collect();
    let rs = &avg_gain / &avg_loss;
    let rsi = rs.mapv(|rs| 100.0 - (100.0 / (1.0 + rs)));
    rsi
}

fn calculate_macd(prices: &Array1<f64>) -> Array1<f64> {
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    ema12 - ema26
}

fn calculate_bollinger_bands(prices: &Array1<f64>, window_size: usize) -> Array2<f64> {
    let sma = calculate_moving_averages(prices, window_size);
    let std = prices.windows(window_size)
        .into_iter()
        .map(|window| window.std(0.0))
        .collect::<Array1<f64>>();
    let upper = &sma + 2.0 * &std;
    let lower = &sma - 2.0 * &std;
    let len = sma.len();
    Array2::from_shape_fn((3, len), |(i, j)| {
        match i {
            0 => upper[j],
            1 => sma[j],
            2 => lower[j],
            _ => unreachable!(),
        }
    })
}

fn calculate_vwap(prices: &Array1<f64>, volumes: &Array1<f64>, window_size: usize) -> Array1<f64> {
    let pv = prices * volumes;
    let vwap = pv.windows(window_size)
        .into_iter()
        .zip(volumes.windows(window_size).into_iter())
        .map(|(pv_window, vol_window)| {
            pv_window.sum() / vol_window.sum()
        })
        .collect();
    vwap
}

fn calculate_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();
    
    let numerator: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum();
    
    let x_std = x.std(0.0);
    let y_std = y.std(0.0);
    
    numerator / (n * x_std * y_std)
} 