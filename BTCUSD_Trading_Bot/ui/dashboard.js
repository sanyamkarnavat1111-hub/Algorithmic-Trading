/**
 * dashboard.js — Position-based trading dashboard.
 *
 * Refresh:
 *   - Price + prediction + position: every 30 seconds
 *   - Chart: every 10 minutes
 */

let priceChart = null;
const CHART_SIZE = 96;

document.addEventListener('DOMContentLoaded', () => {
  loadDashboardData();
  loadChartData();
  setInterval(loadDashboardData, 30000);
  setInterval(loadChartData, 600000);
});

// ── Dashboard data ───────────────────────────────────────────────────────────

async function loadDashboardData() {
  try {
    const res = await fetch('/api/dashboard');
    const data = await res.json();
    if (data.error) return;

    // BTC Price
    if (data.btc_price > 0) {
      document.getElementById('btc-price').textContent =
        '$' + data.btc_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

    const model = data.models[0];
    if (!model) return;

    // Prediction
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
      document.getElementById('pred-direction').textContent = 'Models not ready';
      document.getElementById('pred-direction').className = 'pred-value';
      document.getElementById('pred-confidence').textContent = '—';
      document.getElementById('pred-high').textContent = '—';
      document.getElementById('pred-low').textContent = '—';
      document.getElementById('pred-current').textContent = data.btc_price > 0 ? '$' + data.btc_price.toLocaleString() : '—';
    }

    // Open positions (multiple)
    renderOpenPositions(model.open_positions, pred?.current_price);

    // Stats
    const stats = model.stats || {};
    const pnlEl = document.getElementById('stat-pnl');
    const pnl = stats.total_pnl || 0;
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
    pnlEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('stat-total').textContent = stats.total_trades || 0;
    document.getElementById('stat-wins').textContent = stats.wins || 0;
    document.getElementById('stat-winrate').textContent =
      stats.total_trades > 0 ? (stats.win_rate * 100).toFixed(1) + '%' : '—';

    // Activity log
    renderActivityLog(model.activity_log || []);

    // Model info
    renderModelInfo('model-direction-info', model.direction_model, 'F1 Score');
    renderModelInfo('model-high-info', model.range_high_model, 'MAE');
    renderModelInfo('model-low-info', model.range_low_model, 'MAE');

  } catch (e) {
    console.error('Dashboard error:', e);
  }
}

// ── Open Positions (up to 5) ─────────────────────────────────────────────────

function renderOpenPositions(positions, currentPrice) {
  const el = document.getElementById('open-position');

  if (!positions || positions.length === 0) {
    el.innerHTML = '<div style="color:var(--muted)">No open positions. Waiting for BUY signal...</div>';
    return;
  }

  el.innerHTML = positions.map((pos, i) => {
    const pnlPct = ((currentPrice - pos.entry_price) / pos.entry_price) * 100;
    const pnlDollar = (currentPrice - pos.entry_price) * pos.btc_quantity;
    const pnlColor = pnlPct >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlSign = pnlPct >= 0 ? '+' : '';
    const opened = pos.opened_at ? new Date(pos.opened_at).toLocaleString('en-US', {hour:'2-digit', minute:'2-digit'}) : '—';

    return `
      <div class="position-card" style="margin-bottom:8px">
        <div class="position-grid">
          <div><span class="pos-label">Position #${i+1}</span><span class="pos-value green">BUY</span></div>
          <div><span class="pos-label">Entry</span><span class="pos-value">$${pos.entry_price.toLocaleString()}</span></div>
          <div><span class="pos-label">Amount</span><span class="pos-value">$${pos.amount_usdt.toFixed(0)}</span></div>
          <div><span class="pos-label">Current P&L</span><span class="pos-value" style="color:${pnlColor}">${pnlSign}${pnlPct.toFixed(2)}% (${pnlSign}$${pnlDollar.toFixed(2)})</span></div>
          <div><span class="pos-label">Target (+10%)</span><span class="pos-value green">$${(pos.entry_price * 1.10).toLocaleString()}</span></div>
          <div><span class="pos-label">Stop (-5%)</span><span class="pos-value red">$${(pos.entry_price * 0.95).toLocaleString()}</span></div>
        </div>
      </div>
    `;
  }).join('');
}

// ── Activity Log ─────────────────────────────────────────────────────────────

function renderActivityLog(logs) {
  const container = document.getElementById('activity-log');

  if (!logs || logs.length === 0) {
    container.innerHTML = '<div class="activity-empty">No recent activity found. Waiting for heartbeat...</div>';
    return;
  }

  container.innerHTML = logs.map(log => {
    const timeStr = log.timestamp ? new Date(log.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '';
    
    // Simple coloring for keywords
    let msgHTML = log.message;
    if (msgHTML.includes('BUY')) msgHTML = msgHTML.replace('BUY', '<span style="color:var(--green);font-weight:bold;">BUY</span>');
    if (msgHTML.includes('SELL')) msgHTML = msgHTML.replace('SELL', '<span style="color:var(--red);font-weight:bold;">SELL</span>');
    if (msgHTML.includes('HOLD')) msgHTML = msgHTML.replace('HOLD', '<span style="color:var(--muted);font-weight:bold;">HOLD</span>');
    if (msgHTML.includes('🎯')) msgHTML = '<span style="color:var(--green)">' + msgHTML + '</span>';
    if (msgHTML.includes('❌') || msgHTML.includes('⚠️')) msgHTML = '<span style="color:var(--red)">' + msgHTML + '</span>';

    return `
      <div class="activity-item">
        <div class="activity-time">${timeStr}</div>
        <div class="activity-msg">${msgHTML}</div>
      </div>
    `;
  }).join('');
}

// ── Model info ───────────────────────────────────────────────────────────────

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

// ── Chart ────────────────────────────────────────────────────────────────────

async function loadChartData() {
  try {
    const res = await fetch('/api/candles/15m?limit=' + CHART_SIZE);
    const data = await res.json();
    if (!data.candles || data.candles.length === 0) return;

    const labels = data.candles.map(c => new Date(c.t));
    const prices = data.candles.map(c => c.c);
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
      priceChart.data.labels = labels;
      priceChart.data.datasets[0].data = prices;
      priceChart.update('none');
      return;
    }

    priceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'BTC Close',
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
            ticks: { color: '#64748b', callback: v => '$' + v.toLocaleString(), font: { size: 10 } },
          }
        }
      }
    });
  } catch (e) {
    console.error('Chart error:', e);
  }
}
