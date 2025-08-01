// A high-performance tick analysis library written in Rust.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, Axis, s, stack};
use ndarray_parallel::prelude::*;
use rayon::prelude::*;
use pyo3::types::PyDict;
use statrs::statistics::{Data, Statistics};

/// This module provides high-performance implementations of common financial
/// calculations, exposed to Python via PyO3.
#[pymodule]
fn tick_analysis_processing(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_technical_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_correlation, m)?)?;
    Ok(())
}

/// Calculate multiple technical indicators for the given price and volume data.
/// Returns a dictionary of NumPy arrays.
#[pyfunction]
fn calculate_technical_indicators<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    window_size: usize,
) -> PyResult<&'py PyDict> {
    let prices_arr = prices.as_array();
    let volumes_arr = volumes.as_array();

    // --- FIX 2: Add length validation between input arrays. ---
    if prices_arr.len() != volumes_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input 'prices' and 'volumes' arrays must have the same length."
        ));
    }

    if prices_arr.len() < window_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input prices length cannot be less than window_size.",
        ));
    }

    // --- FIX 1: Replaced `join3` with nested `join` calls. ---
    let (group1, group2) = rayon::join(
        || { // First group of 3 tasks
            let (sma, (ema, rsi)) = rayon::join(
                || calculate_sma(&prices_arr, window_size),
                || rayon::join(
                    || calculate_ema(&prices_arr, 12),
                    || calculate_rsi(&prices_arr, window_size)
                )
            );
            (sma, ema, rsi)
        },
        || { // Second group of 3 tasks
            let (macd_tuple, (bb, vwap)) = rayon::join(
                || calculate_macd(&prices_arr),
                || rayon::join(
                    || calculate_bollinger_bands(&prices_arr, window_size),
                    || calculate_vwap(&prices_arr, &volumes_arr, window_size)
                )
            );
            (macd_tuple, bb, vwap)
        }
    );

    let (sma, ema, rsi) = group1;
    let (macd_tuple, bb, vwap) = group2;
    let (macd, macd_signal) = macd_tuple;


    let result = PyDict::new(py);
    result.set_item("sma", sma.to_pyarray(py))?;
    result.set_item("ema", ema.to_pyarray(py))?;
    result.set_item("rsi", rsi.to_pyarray(py))?;
    result.set_item("macd", macd.to_pyarray(py))?;
    result.set_item("macd_signal", macd_signal.to_pyarray(py))?;
    result.set_item("bollinger_bands", bb.to_pyarray(py))?;
    result.set_item("vwap", vwap.to_pyarray(py))?;
    Ok(result)
}

/// Calculate basic descriptive statistics over a given axis.
#[pyfunction]
fn calculate_statistics<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    axis: usize,
) -> PyResult<&'py PyDict> {
    let data_arr = data.as_array();

    // --- FIX 1: Replaced `join4` with nested `join` calls. ---
    let ((mean, std), (min, max)) = rayon::join(
        || rayon::join(
            || data_arr.mean_axis(Axis(axis)),
            || Some(data_arr.std_axis(Axis(axis), 0.0))
        ),
        || rayon::join(
            || data_arr.map_axis(Axis(axis), |row| row.min().unwrap_or(f64::NAN)).into_owned(),
            || data_arr.map_axis(Axis(axis), |row| row.max().unwrap_or(f64::NAN)).into_owned()
        )
    );

    let result = PyDict::new(py);
    let mean_arr = mean.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Cannot compute mean on empty data."))?;
    let std_arr = std.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Cannot compute std dev on empty data."))?;
    
    result.set_item("mean", mean_arr.to_pyarray(py))?;
    result.set_item("std", std_arr.to_pyarray(py))?;
    result.set_item("min", min.to_pyarray(py))?;
    result.set_item("max", max.to_pyarray(py))?;
    Ok(result)
}

/// Calculate the Pearson correlation matrix for a 2D array of asset prices.
#[pyfunction]
fn calculate_correlation(data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let data_arr = data.as_array();
    let p = data_arr.ncols();
    let mut correlation = Array2::from_elem((p, p), 1.0);

    let mut pairs = Vec::new();
    for i in 0..p {
        for j in (i + 1)..p {
            pairs.push((i, j));
        }
    }

    let results: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let col_i = data_arr.column(i);
            let col_j = data_arr.column(j);
            let corr = calculate_pearson_correlation(&col_i.to_owned(), &col_j.to_owned());
            ((i, j), corr)
        })
        .collect();

    for ((i, j), corr_value) in results {
        correlation[[i, j]] = corr_value;
        correlation[[j, i]] = corr_value;
    }
    
    Python::with_gil(|py| Ok(correlation.to_pyarray(py).to_owned()))
}

// --- Helper Functions: All indicators now return same-length padded arrays ---

fn pad_indicator(data: &Array1<f64>, result: Array1<f64>) -> Array1<f64> {
    let data_len = data.len();
    let result_len = result.len();
    if data_len > result_len {
        let pad_len = data_len - result_len;
        let mut padded_result = Array1::from_elem(data_len, f64::NAN);
        padded_result.slice_mut(s![pad_len..]).assign(&result);
        padded_result
    } else {
        result
    }
}

fn calculate_sma(prices: &Array1<f64>, window: usize) -> Array1<f64> {
    let sma: Array1<f64> = prices.windows(window)
        .into_iter()
        .map(|w| w.mean().unwrap_or(f64::NAN))
        .collect();
    pad_indicator(prices, sma)
}

fn calculate_ema(prices: &Array1<f64>, window: usize) -> Array1<f64> {
    let alpha = 2.0 / (window as f64 + 1.0);
    let mut ema = Array1::from_elem(prices.len(), f64::NAN);
    if prices.is_empty() { return ema; }

    ema[0] = prices[0];
    for i in 1..prices.len() {
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
    }
    ema
}

fn calculate_rsi(prices: &Array1<f64>, window: usize) -> Array1<f64> {
    if prices.len() <= window { return Array1::from_elem(prices.len(), f64::NAN); }
    let deltas = prices.slice(s![1..]).to_owned() - prices.slice(s![..-1]);
    let mut gains = Array1::zeros(deltas.len());
    let mut losses = Array1::zeros(deltas.len());

    for i in 0..deltas.len() {
        if deltas[i] > 0.0 {
            gains[i] = deltas[i];
        } else {
            losses[i] = -deltas[i];
        }
    }
    
    let mut avg_gain = gains.slice(s![..window]).mean().unwrap_or(0.0);
    let mut avg_loss = losses.slice(s![..window]).mean().unwrap_or(0.0);
    
    let mut rsi_values = Vec::with_capacity(prices.len() - window -1);
    
    let first_rs = if avg_loss == 0.0 { 100.0 } else { avg_gain / avg_loss };
    rsi_values.push(100.0 - (100.0 / (1.0 + first_rs)));

    for i in window..deltas.len() {
        avg_gain = (avg_gain * (window as f64 - 1.0) + gains[i]) / window as f64;
        avg_loss = (avg_loss * (window as f64 - 1.0) + losses[i]) / window as f64;
        
        let rs = if avg_loss == 0.0 { 100.0 } else { avg_gain / avg_loss };
        rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
    }
    
    pad_indicator(&prices, Array1::from(rsi_values))
}

fn calculate_macd(prices: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    let macd_line = ema12 - ema26;
    let signal_line = calculate_ema(&macd_line, 9);
    (macd_line, signal_line)
}

fn calculate_bollinger_bands(prices: &Array1<f64>, window: usize) -> Array2<f64> {
    let sma = calculate_sma(prices, window);
    let std_dev_values: Array1<f64> = prices.windows(window)
        .into_iter()
        .map(|w| w.std(0.0))
        .collect();
    let std_dev = pad_indicator(prices, std_dev_values);

    let upper = &sma + 2.0 * &std_dev;
    let lower = &sma - 2.0 * &std_dev;
    
    stack![Axis(0), upper, sma, lower]
}

fn calculate_vwap(prices: &Array1<f64>, volumes: &Array1<f64>, window: usize) -> Array1<f64> {
    let typical_price_volume = prices * volumes;
    let vwap_values: Array1<f64> = typical_price_volume.windows(window)
        .into_iter()
        .zip(volumes.windows(window))
        .map(|(pv_window, vol_window)| {
            let vol_sum = vol_window.sum();
            if vol_sum == 0.0 { f64::NAN } else { pv_window.sum() / vol_sum }
        })
        .collect();
    pad_indicator(prices, vwap_values)
}

fn calculate_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    if x.len() != y.len() || x.len() == 0 { return f64::NAN; }
    
    let data_x = Data::new(x.as_slice().unwrap());
    let data_y = Data::new(y.as_slice().unwrap());

    let cov = data_x.covariance(&data_y);
    let std_x = data_x.std_dev();
    let std_y = data_y.std_dev();
    
    if std_x.is_none() || std_y.is_none() || std_x.unwrap() == 0.0 || std_y.unwrap() == 0.0 {
        return f64::NAN;
    }
    
    cov / (std_x.unwrap() * std_y.unwrap())
}
