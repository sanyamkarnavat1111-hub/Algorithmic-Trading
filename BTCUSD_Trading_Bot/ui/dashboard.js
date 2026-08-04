/**
 * dashboard.js — Fetches API data and updates the UI every 30 seconds.
 *
 * No frameworks. Plain JavaScript.
 * Easy to read: one function per UI section.
 */

const REFRESH_MS  = 30_000;  // refresh every 30 seconds
const MODELS      = ['ai_15m'];


let priceChart       = null;
let currentTimeframe = '15m';
let allTrades        = [];   // cached for filtering without re-fetch

// ── Bootstrap ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  loadChart(currentTimeframe);
  setInterval(loadDashboard, REFRESH_MS);
  setInterval(() => loadChart(currentTimeframe), REFRESH_MS);
});

// ── Main Data Loader ───────────────────────────────────────────────────────

async function loadDashboard() {
  try {
    const res  = await fetch('/api/dashboard');
    const data = await res.json();

    updateBtcPrice(data.btc_price);
    updateLastUpdate();

    data.models.forEach(model => {
      updateModelCard(model);
    });

    // Collect all recent trades for the table
    allTrades = [];
    data.models.forEach(model => {
      model.recent_trades.forEach(t => {
        allTrades.push({ ...t, model_id: model.model_id });
      });
      model.open_trades.forEach(t => {
        allTrades.push({ ...t, model_id: model.model_id, status: 'OPEN' });
      });
    });

    renderTradesTable(allTrades);

  } catch (err) {
    console.error('Dashboard load error:', err);
  }
}

// ── BTC Price ──────────────────────────────────────────────────────────────

function updateBtcPrice(price) {
  const el = document.getElementById('btc-price');
  if (price && price > 0) {
    el.textContent = '$' + price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
}

function updateLastUpdate() {
  document.getElementById('last-update').textContent =
    new Date().toLocaleTimeString();
}

// ── Model Card Updates ─────────────────────────────────────────────────────

function updateModelCard(model) {
  const id = model.model_id;  // e.g., 'ai_1h'

  // Signal badge
  const signalEl = document.getElementById(`signal-${id}`);
  const signal   = model.prediction?.signal ?? '—';
  signalEl.textContent = signal;
  signalEl.className   = `signal-badge ${signal}`;

  // Confidence
  const conf = model.prediction?.confidence;
  document.getElementById(`conf-${id}`).textContent =
    conf != null ? (conf * 100).toFixed(1) + '%' : '—';

  // P&L
  const pnlEl = document.getElementById(`pnl-${id}`);
  const pnl   = model.stats?.total_pnl ?? null;
  if (pnl != null) {
    pnlEl.textContent  = (pnl >= 0 ? '+' : '') + pnl.toFixed(0) + ' $';
    pnlEl.className    = 'stat-value ' + (pnl >= 0 ? 'positive' : 'negative');
  } else {
    pnlEl.textContent = '—';
  }

  // Win rate
  const wr = model.stats?.win_rate;
  document.getElementById(`winrate-${id}`).textContent =
    wr != null ? (wr * 100).toFixed(1) + '%' : '—';

  // Model version
  document.getElementById(`version-${id}`).textContent =
    model.version ? `v${model.version}` : 'Untrained';

  // Probability bars
  const probs = model.prediction?.probabilities;
  if (probs) {
    setProbBar(id, 'sell', probs.SELL);
    setProbBar(id, 'hold', probs.HOLD);
    setProbBar(id, 'buy',  probs.BUY);
  }

  // Open trades
  renderOpenTrade(id, model.open_trades);
}

function setProbBar(modelId, type, value) {
  const pct = Math.round((value ?? 0) * 100);
  document.getElementById(`pb-${type}-${modelId}`).style.width  = pct + '%';
  document.getElementById(`pv-${type}-${modelId}`).textContent   = pct + '%';
}

function renderOpenTrade(modelId, openTrades) {
  const container = document.getElementById(`open-trade-${modelId}`);
  if (!openTrades || openTrades.length === 0) {
    container.innerHTML = '';
    return;
  }

  const t    = openTrades[0];
  const cls  = t.signal === 'BUY' ? 'buy-trade' : 'sell-trade';
  const sign = t.signal === 'BUY' ? '📈' : '📉';

  container.innerHTML = `
    <div class="open-trade-badge ${cls}">
      <span>${sign} Open ${t.signal} @ $${parseFloat(t.entry_price).toLocaleString()}</span>
      <span style="color: var(--text-muted)">SL: $${parseFloat(t.stop_loss).toLocaleString()}</span>
    </div>
  `;
}

// ── Trades Table ───────────────────────────────────────────────────────────

function renderTradesTable(trades) {
  const tbody = document.getElementById('trades-body');

  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-row">No trades yet — waiting for models to train and signal...</td></tr>';
    return;
  }

  // Sort newest first
  const sorted = [...trades].sort((a, b) => {
    const ta = new Date(a.opened_at || 0);
    const tb = new Date(b.opened_at || 0);
    return tb - ta;
  });

  tbody.innerHTML = sorted.map(t => {
    const pnl    = t.pnl;
    const pnlPct = t.pnl_pct;
    const pnlStr = pnl != null
      ? `<span class="${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>`
      : '—';
    const pnlPctStr = pnlPct != null
      ? `<span class="${pnlPct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${(pnlPct * 100).toFixed(2)}%</span>`
      : '—';

    const statusBadge = {
      'OPEN':        '<span class="badge badge-open">OPEN</span>',
      'CLOSED':      '<span class="badge badge-closed">CLOSED</span>',
      'STOPPED_OUT': '<span class="badge badge-stopped">STOPPED</span>',
    }[t.status] ?? t.status;

    const signalBadge = {
      'BUY':  '<span class="badge badge-buy">BUY</span>',
      'SELL': '<span class="badge badge-sell">SELL</span>',
      'HOLD': '<span class="badge badge-hold">HOLD</span>',
    }[t.signal] ?? t.signal;

    const modelLabel = { ai_15m: 'AI-15M', ai_1h: 'AI-1H', ai_8h: 'AI-8H', ai_1d: 'AI-1D' }[t.model_id] ?? t.model_id;

    const timeStr = t.opened_at
      ? new Date(t.opened_at).toLocaleString()
      : '—';

    return `
      <tr class="trade-row" data-model="${t.model_id}">
        <td>${modelLabel}</td>
        <td>${signalBadge}</td>
        <td>$${parseFloat(t.entry_price || 0).toLocaleString()}</td>
        <td>${t.exit_price ? '$' + parseFloat(t.exit_price).toLocaleString() : '—'}</td>
        <td>${pnlStr}</td>
        <td>${pnlPctStr}</td>
        <td>${t.confidence ? (t.confidence * 100).toFixed(1) + '%' : '—'}</td>
        <td>${statusBadge}</td>
        <td>${timeStr}</td>
      </tr>
    `;
  }).join('');
}

// ── Trade Filter Buttons ───────────────────────────────────────────────────

function filterTrades(modelId, btn) {
  // Update active button
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (modelId === 'all') {
    renderTradesTable(allTrades);
  } else {
    renderTradesTable(allTrades.filter(t => t.model_id === modelId));
  }
}

// ── Chart ──────────────────────────────────────────────────────────────────

async function loadChart(timeframe) {
  try {
    const res  = await fetch(`/api/candles/${timeframe}?limit=100`);
    const data = await res.json();

    if (!data.candles || data.candles.length === 0) {
      console.log('No candle data for chart yet');
      return;
    }

    renderCandlestickChart(data.candles);
  } catch (err) {
    console.error('Chart load error:', err);
  }
}

function renderCandlestickChart(candles) {
  const ctx = document.getElementById('priceChart').getContext('2d');

  // Format for Chart.js Financial
  const ohlcData = candles.map(c => ({
    x: new Date(c.t).getTime(),
    o: c.o,
    h: c.h,
    l: c.l,
    c: c.c,
  }));

  if (priceChart) {
    priceChart.data.datasets[0].data = ohlcData;
    priceChart.update('none');
    return;
  }

  priceChart = new Chart(ctx, {
    type: 'candlestick',
    data: {
      datasets: [{
        label:           'BTC/USDT',
        data:            ohlcData,
        color: {
          up:   '#22c55e',
          down: '#ef4444',
          unchanged: '#64748b',
        },
        borderColor: {
          up:   '#22c55e',
          down: '#ef4444',
          unchanged: '#64748b',
        },
      }]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              return [`O: $${d.o.toLocaleString()}`, `H: $${d.h.toLocaleString()}`,
                      `L: $${d.l.toLocaleString()}`, `C: $${d.c.toLocaleString()}`];
            }
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          grid:  { color: '#1c2030' },
          ticks: { color: '#64748b', maxTicksLimit: 10 },
        },
        y: {
          position: 'right',
          grid:     { color: '#1c2030' },
          ticks:    { color: '#64748b', callback: v => '$' + v.toLocaleString() },
        }
      }
    }
  });
}

// ── Chart Timeframe Switcher ───────────────────────────────────────────────

function switchChart(timeframe, btn) {
  currentTimeframe = timeframe;
  document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  // Destroy and reload chart for new timeframe
  if (priceChart) {
    priceChart.destroy();
    priceChart = null;
  }
  loadChart(timeframe);
}
