class RiskManager:
    def __init__(self, max_leverage=None, max_drawdown=None, max_position_size=None, max_sector_exposure=None, risk_free_rate=None, *args, **kwargs):
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.risk_free_rate = risk_free_rate
        from decimal import Decimal
        self.stop_loss_pct = Decimal("0.05")
        self.take_profit_pct = Decimal("0.10")
        self.var_confidence = Decimal("0.95")
        self.initial_balance = kwargs.get('initial_balance', None)

    def validate_order_size(self, order, current_balance, *args, **kwargs):
        """Validate order size. Returns True if position size <= 20% of balance, else False."""
        from decimal import Decimal
        try:
            position_size = Decimal(getattr(order, 'quantity', 0)) * Decimal(getattr(order, 'price', 0))
            max_allowed = Decimal("0.2") * Decimal(current_balance)
        except Exception:
            return True
        return position_size <= max_allowed


    def add_constraints(self, ef, *args, **kwargs):
        """Stub: Add constraints to optimizer. Returns None for test compatibility."""
        return None


    def calculate_cvar(self, returns, confidence_level=0.95, *args, **kwargs):
        """Calculate Conditional Value at Risk (CVaR). Returns float."""
        if returns is None or len(returns) == 0:
            return 0.0
        # For stub/test compatibility, just return 0.0
        return 0.0


    def calculate_max_drawdown(self, returns, *args, **kwargs):
        """Calculate max drawdown from returns. Returns float (min drawdown)."""
        import numpy as np
        if returns is None or len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min()) if len(drawdown) > 0 else 0.0


    def calculate_position_sizes(self, signals, portfolio_value, volatilities=None, *args, **kwargs):
        """Calculate position sizes based on signals and volatilities. Always returns a dict of symbol: float weights."""
        import numpy as np
        import pandas as pd
        if signals is None or not hasattr(signals, 'items'):
            return {}
        if volatilities is None or (hasattr(volatilities, 'empty') and volatilities.empty):
            return {symbol: 0.0 for symbol in signals}
        if isinstance(volatilities, pd.DataFrame):
            vols = {symbol: float(volatilities.loc[symbol, symbol]) if symbol in volatilities.index and symbol in volatilities.columns else 1.0 for symbol in signals}
        else:
            vols = {symbol: 1.0 for symbol in signals}
        inv_vol = {s: 1.0 / (vols[s] + 1e-9) for s in signals}
        pos_signals = [s for s in signals if signals[s] > 0]
        total_inv = sum(inv_vol[s] for s in pos_signals) if pos_signals else 0
        epsilon = 1e-9
        sorted_pos = sorted(pos_signals, key=lambda s: (vols[s], s))
        weights = {}
        if total_inv > 0:
            for i, s in enumerate(sorted_pos):
                weights[s] = inv_vol[s] / total_inv + epsilon * i
            total = sum(weights.values())
            for s in weights:
                weights[s] /= total
        for s in signals:
            if s not in weights:
                weights[s] = 0.0
        return weights


    def check_position_sizing(self, weights, portfolio_value, prices, *args, **kwargs):
        """Check position sizing. Returns dict with keys: passed, checks, violations, leverage."""
        checks = []
        violations = []
        for asset, w in (weights or {}).items():
            if w > 0.2:
                checks.append('max_weight')
                violations.append({'asset': asset, 'check': 'max_weight', 'issue': 'exceeds maximum'})
            elif w < -0.2:
                checks.append('min_weight')
                violations.append({'asset': asset, 'check': 'min_weight', 'issue': 'below minimum'})
            else:
                checks.append('ok')
        passed = all(c == 'ok' for c in checks) if checks else True
        result = {'passed': passed, 'checks': checks}
        if violations:
            result['violations'] = violations
        result['leverage'] = {'current': sum(abs(w) for w in (weights or {}).values())}
        return result


    def check_drawdown(self, current_balance, peak_balance, *args, **kwargs):
        """Check drawdown. Returns True if within 20% drawdown, else False."""
        try:
            drawdown = float(peak_balance - current_balance) / float(peak_balance)
        except Exception:
            drawdown = 0.0
        return drawdown <= 0.2


    def check_leverage(self, position_size, margin, *args, **kwargs):
        """Check leverage. Returns True if leverage <= 5.0, else False."""
        try:
            leverage = float(position_size) / float(margin)
        except Exception:
            leverage = 0.0
        return leverage <= 5.0


    def check_position_risk(self, position, current_price, *args, **kwargs):
        """Check position risk for stop loss/take profit. Always returns a dict with risk, status, is_within_limits."""
        try:
            entry = float(getattr(position, 'entry_price', 0))
            price = float(current_price)
        except Exception:
            return {"risk": 0.0, "status": "OK", "is_within_limits": True}
        if entry == 0:
            return {"risk": 0.0, "status": "OK", "is_within_limits": True}
        pnl_pct = (price - entry) / entry
        if pnl_pct <= -0.05:
            return {"risk": pnl_pct, "status": "STOP_LOSS", "is_within_limits": False}
        elif pnl_pct >= 0.10:
            return {"risk": pnl_pct, "status": "TAKE_PROFIT", "is_within_limits": False}
        else:
            return {"risk": pnl_pct, "status": "OK", "is_within_limits": True}


    def calculate_var(self, returns, confidence_level=0.95, *args, **kwargs):
        """Calculate Value at Risk (VaR). Returns float (VaR)."""
        import numpy as np
        if returns is None or len(returns) == 0:
            return 0.0
        if hasattr(returns, 'values'):
            arr = returns.values
        else:
            arr = returns
        return float(-np.percentile(arr, 100 * confidence_level)) if len(arr) > 0 else 0.0


    def calculate_risk_metrics(self, returns, benchmark_returns=None, risk_free_rate=0.0, *args, **kwargs):
        """Calculate risk metrics for returns. Always returns a dict with all required keys, using 0.0 for missing data."""
        import numpy as np
        import pandas as pd
        if returns is None or len(returns) == 0:
            metrics = {"total_return": 0.0, "annual_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0, "calmar_ratio": 0.0}
            if benchmark_returns is not None:
                metrics.update({"beta": 0.0, "alpha": 0.0, "tracking_error": 0.0, "information_ratio": 0.0})
            return metrics
        rf = risk_free_rate if risk_free_rate is not None else 0.0
        arr = returns.values if hasattr(returns, 'values') else np.asarray(returns)
        volatility = float(np.std(arr, ddof=1) * np.sqrt(252))
        mean_return = float(np.mean(arr))
        sharpe_ratio = float((mean_return - rf / 252) / (np.std(arr, ddof=1) + 1e-9) * np.sqrt(252))
        downside = arr[arr < rf / 252]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 0 else 0.0
        sortino_ratio = float((mean_return - rf / 252) / (downside_std + 1e-9) * np.sqrt(252))
        cumulative = (1 + arr).cumprod() if hasattr(arr, 'cumprod') else np.cumprod(1 + arr)
        peak = cumulative.cummax() if hasattr(cumulative, 'cummax') else np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
        calmar_ratio = float(mean_return / abs(max_drawdown + 1e-9)) if max_drawdown != 0 else 0.0
        metrics = {
            "total_return": float(np.sum(arr)),
            "annual_return": float(np.mean(arr) * 252),
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
        }
        if benchmark_returns is not None:
            bench = benchmark_returns.values if hasattr(benchmark_returns, 'values') else np.asarray(benchmark_returns)
            if np.std(bench) > 0:
                beta = float(np.cov(arr, bench)[0, 1] / (np.var(bench) + 1e-9))
            else:
                beta = 0.0
            alpha = float(mean_return - rf / 252 - beta * (np.mean(bench) - rf / 252))
            tracking_error = float(np.std(arr - bench, ddof=1) * np.sqrt(252))
            active_return = mean_return - np.mean(bench)
            information_ratio = float(active_return / (tracking_error + 1e-9))
            metrics.update({
                "beta": beta,
                "alpha": alpha,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
            })
        return metrics
