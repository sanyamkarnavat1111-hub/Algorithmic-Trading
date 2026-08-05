/**
 * dashboard.js — Dashboard with smart refresh intervals.
 *
 * Refresh strategy:
 *   - BTC price + prediction + portfolio: every 30 seconds
 *   - Price chart: every 10 minutes (candles only change every 15 min)
 *   - Chart uses sliding window: drops oldest, appends newest
 */

let priceChart = null;
let chartCandles = []; // in-memory candle buffer (sliding window)
const CHART_SIZE = 96; // 96 candles = 24 hours of 15-min data

// ── Startup ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Load everything immediately
  loadDashboardData();
  loadChartData();

  // Price, prediction, portfolio — refresh every 30 seconds
  setInterval(loadDashboardData, 30000);

  // Chart — refresh every 10 minutes (600,000 ms)
  setInterval(loadChartData, 600000);
});

// ── Dashboard data (price, prediction, portfolio, trades) ────────────────────

async function loadDashboardData() {
  try {
    const res = await fetch('/api/dashboard');
    const data = await res.json();
    if (data.error) return;

    // BTC Price (top bar)
    if (data.btc_price > 0) {
      document.getElementById('btc-price').textContent =
        '$' + data.btc_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }

    // Last update time
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

    const model = data.models[0];
    if (!model) return;

    // Prediction section
    const pred = model.prediction;
    if (pred) {
      const dirEl = document.getElementById('pred-direction');
      dirEl.textContent = pred.direction;
      dirEl.className = 'pred-value ' + pred.direction.toLowerCase();

      document.getElementById('pred-confidence').textContent = (pred.confidence * 100).toFixed(1) + '%';
      document.getElementById('pred-high').textContent = '$' + pred.predicted_high.toLocaleString();
      document.getElementById('pred-low').textContent = '$' + pred.predicted_low.toLocaleString();
      document.getElementById('pred-current').textContent = '$' + pred.current_price.toLocaleString();
    } else {
      document.getElementById('pred-direction').textContent = 'Not ready';
      document.getElementById('pred-direction').className = 'pred-value';
      document.getElementById('pred-confidence').textContent = '—';
      document.getElementById('pred-high').textContent = '—';
      document.getElementById('pred-low').textContent = '—';
      document.getElementById('pred-current').textContent = data.btc_price > 0 ? '$' + data.btc_price.toLocaleString() : '—';
      document.getElementById('pred-action').textContent = 'Waiting for models';
      document.getElementById('pred-action').className = 'pred-value';
    }

    // Portfolio
    if (data.portfolio) {
      document.getElementById('port-usdt').textContent =
        '$' + parseFloat(data.portfolio.usdt_balance).toLocaleString('en-US', {maximumFractionDigits: 0});
      document.getElementById('port-btc').textContent =
        parseFloat(data.portfolio.btc_quantity).toFixed(6) + ' BTC';
      const avg = parseFloat(data.portfolio.btc_avg_price);
      document.getElementById('port-avg').textContent = avg > 0 ? '$' + avg.toLocaleString() : '—';
    }

    // Total P&L
    const pnl = model.stats?.total_pnl || 0;
    const pnlEl = document.getElementById('port-pnl');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
    pnlEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';

    // Latest action
    if (model.recent_trades && model.recent_trades.length > 0) {
      const latest = model.recent_trades[0];
      const actionEl = document.getElementById('pred-action');
      actionEl.textContent = latest.signal;
      actionEl.className = 'pred-value ' + latest.signal.toLowerCase();
    }

    // Trade history table
    renderTrades(model.recent_trades || []);

    // Model info
    renderModelInfo('model-direction-info', model.direction_model, 'F1 Score');
    renderModelInfo('model-high-info', model.range_high_model, 'MAE');
    renderModelInfo('model-low-info', model.range_low_model, 'MAE');

  } catch (e) {
    console.error('Dashboard load error:', e);
  }
}

// ── Chart data (refreshes every 10 min, uses sliding window) ─────────────────

async function loadChartData() {
  try {
    const res = await fetch('/api/candles/15m?limit=' + CHART_SIZE);
    const data = await res.json();

    if (!data.candles || data.candles.length === 0) return;

    // Replace buffer with fresh data from API
    chartCandles = data.candles;

    renderChart();
  } catch (e) {
    console.error('Chart load error:', e);
  }
}

function renderChart() {
  if (chartCandles.length === 0) return;

  const labels = chartCandles.map(c => new Date(c.t));
  const prices = chartCandles.map(c => c.c);

  const ctx = document.getElementById('priceChart').getContext('2d');

  if (priceChart) {
    // Update existing chart (no destroy/recreate — smooth)
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = prices;
    priceChart.update('none');
    return;
  }

  // Create chart first time
  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'BTC Close Price',
        data: prices,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'time',
          grid: { color: '#1a1d2a' },
          ticks: { color: '#64748b', maxTicksLimit: 8, font: { size: 10 } },
        },
        y: {
          position: 'right',
          grid: { color: '#1a1d2a' },
          ticks: {
            color: '#64748b',
            callback: v => '$' + v.toLocaleString(),
            font: { size: 10 },
          },
        }
      }
    }
  });
}

// ── Trade history table ──────────────────────────────────────────────────────

function renderTrades(trades) {
  const tbody = document.getElementById('trades-body');

  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No trades yet. Waiting for AI to make decisions...</td></tr>';
    return;
  }

  tbody.innerHTML = trades.map(t => {
    const action = t.signal || '—';
    const actionClass = action === 'BUY' ? 'action-buy' :
                        action === 'SELL' ? 'action-sell' : 'action-hold';
    const price = t.entry_price ? '$' + parseFloat(t.entry_price).toLocaleString() : '—';
    const amount = t.amount > 0 ? '$' + parseFloat(t.amount).toFixed(0) : '—';
    const predHigh = t.predicted_high ? '$' + parseFloat(t.predicted_high).toLocaleString() : '—';
    const predLow = t.predicted_low ? '$' + parseFloat(t.predicted_low).toLocaleString() : '—';
    const pnl = t.pnl && t.pnl !== 0 ? (t.pnl >= 0 ? '+' : '') + '$' + parseFloat(t.pnl).toFixed(2) : '—';
    const pnlClass = t.pnl > 0 ? 'pnl-pos' : t.pnl < 0 ? 'pnl-neg' : '';
    const time = t.opened_at
      ? new Date(t.opened_at).toLocaleString('en-US', {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'})
      : '—';

    return `<tr>
      <td>${time}</td>
      <td class="${actionClass}">${action}</td>
      <td>${price}</td>
      <td>${amount}</td>
      <td>${predHigh}</td>
      <td>${predLow}</td>
      <td class="${pnlClass}">${pnl}</td>
    </tr>`;
  }).join('');
}

// ── Model info cards ─────────────────────────────────────────────────────────

function renderModelInfo(elementId, info, metricName) {
  const el = document.getElementById(elementId);
  if (!info || !info.version) {
    el.innerHTML = '<span style="color:var(--red)">NOT TRAINED</span>';
    return;
  }
  el.innerHTML = `
    <div>Version: <b>v${info.version}</b></div>
    <div>${metricName}: <b>${info.accuracy || '—'}</b></div>
    <div>Rows: <b>${info.train_rows ? info.train_rows.toLocaleString() : '—'}</b></div>
    <div>Trained: <b>${info.trained_at ? new Date(info.trained_at).toLocaleDateString() : '—'}</b></div>
  `;
}
