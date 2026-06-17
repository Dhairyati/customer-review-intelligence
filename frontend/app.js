/**
 * app.js — Customer Review Intelligence System
 *
 * Three page-init functions, one per HTML page:
 *   initSinglePage()    → index.html
 *   initBatchPage()     → batch.html
 *   initAnalyticsPage() → analytics.html
 */

'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────

const API = {
  health:       '/health',
  predict:      '/predict',
  predictBatch: '/predict/batch',
  analytics:    '/analytics',
};

// Confidence thresholds — mirror the backend default (0.65) for the medium band
const CONF_HIGH   = 0.80;
const CONF_MEDIUM = 0.65;

// ── Shared helpers ────────────────────────────────────────────────────────────

/** Check /health and update a banner. Returns true when model is ready. */
async function checkHealth(bannerEl) {
  try {
    const res  = await fetch(API.health);
    const data = await res.json();
    if (!res.ok) {
      showBanner(bannerEl, 'warning', `Model not ready: ${data.detail ?? 'unknown error'}`);
      return false;
    }
    bannerEl.classList.remove('visible');
    return true;
  } catch {
    showBanner(bannerEl, 'warning', 'Cannot reach server. Is it running?');
    return false;
  }
}

function showBanner(el, type, msg) {
  el.className      = `banner ${type} visible`;
  el.textContent    = msg;
}

function hideBanner(el) {
  el.classList.remove('visible');
}

/**
 * Return a CSS class name based on confidence value.
 * conf-high ≥ 80% | conf-medium ≥ 65% | conf-low < 65%
 */
function confClass(confidence) {
  if (confidence >= CONF_HIGH)   return 'conf-high';
  if (confidence >= CONF_MEDIUM) return 'conf-medium';
  return 'conf-low';
}

/** Format confidence as a coloured percentage string (returns HTML). */
function confHtml(confidence) {
  const cls = confClass(confidence);
  return `<span class="${cls}">${(confidence * 100).toFixed(1)}%</span>`;
}

/** Build the three coloured score bars. */
function buildScoreBars(scores) {
  return ['Positive', 'Neutral', 'Negative'].map(cls => {
    const p = (scores[cls] * 100).toFixed(1);
    return `
      <div class="score-row">
        <span class="score-label">${cls}</span>
        <div class="score-bar-track">
          <div class="score-bar-fill ${cls}" style="width:${p}%"></div>
        </div>
        <span class="score-value">${p}%</span>
      </div>`;
  }).join('');
}

/** Format a Date as a readable local timestamp. */
function fmtTimestamp(date) {
  return date.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** Show/hide a spinner inside a button. */
function setSpinner(btnEl, labelEl, spinnerEl, loading, idleLabel) {
  btnEl.disabled          = loading;
  labelEl.textContent     = loading ? 'Analysing…' : idleLabel;
  spinnerEl.style.display = loading ? 'inline-block' : 'none';
}

/** Show/hide an inline field-level error message. */
function setFieldError(el, visible) {
  el.classList.toggle('visible', visible);
}

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Single Review Page ────────────────────────────────────────────────────────

function initSinglePage() {
  const banner         = document.getElementById('model-banner');
  const input          = document.getElementById('review-input');
  const inputError     = document.getElementById('input-error');
  const predictBtn     = document.getElementById('predict-btn');
  const clearBtn       = document.getElementById('clear-btn');
  const charCount      = document.getElementById('char-count');
  const btnLabel       = document.getElementById('btn-label');
  const btnSpinner     = document.getElementById('btn-spinner');
  const resultLoading  = document.getElementById('result-loading');
  const resultCard     = document.getElementById('result-card');
  const resultPill     = document.getElementById('result-pill');
  const resultConf     = document.getElementById('result-confidence');
  const uncertainBadge = document.getElementById('uncertain-badge');
  const scoreBars      = document.getElementById('score-bars');
  const resultTs       = document.getElementById('result-timestamp');

  let modelReady = false;

  // Health check — enable button once model is ready
  (async () => {
    modelReady = await checkHealth(banner);
    if (modelReady && input.value.trim().length > 0) {
      predictBtn.disabled = false;
    }
  })();

  // Character counter + validation
  input.addEventListener('input', () => {
    const len = input.value.length;
    charCount.textContent = len.toLocaleString();
    const hasText = input.value.trim().length > 0;
    predictBtn.disabled = !hasText || !modelReady;
    if (hasText) {
      input.classList.remove('input-error');
      setFieldError(inputError, false);
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    charCount.textContent = '0';
    predictBtn.disabled = true;
    resultCard.className = 'result-card';
    resultLoading.classList.remove('visible');
    input.classList.remove('input-error');
    setFieldError(inputError, false);
    hideBanner(banner);
  });

  predictBtn.addEventListener('click', async () => {
    const text = input.value.trim();

    // ── Input validation
    if (!text) {
      input.classList.add('input-error');
      setFieldError(inputError, true);
      input.focus();
      return;
    }

    input.classList.remove('input-error');
    setFieldError(inputError, false);

    // ── Loading state
    setSpinner(predictBtn, btnLabel, btnSpinner, true, 'Analyse');
    resultLoading.classList.add('visible');
    resultCard.className = 'result-card'; // hide previous result
    hideBanner(banner);

    try {
      const res = await fetch(API.predict, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text }),
      });

      if (!res.ok) {
        const err = await res.json();
        showBanner(banner, 'error', `Error: ${err.detail ?? res.statusText}`);
        return;
      }

      renderSingleResult(await res.json());

    } catch {
      showBanner(banner, 'error', 'Network error. Please try again.');
    } finally {
      resultLoading.classList.remove('visible');
      setSpinner(predictBtn, btnLabel, btnSpinner, false, 'Analyse');
      predictBtn.disabled = false;
    }
  });

  function renderSingleResult(data) {
    // Card colour
    resultCard.className = `result-card label-${data.label} visible`;

    // Label pill
    resultPill.className   = `label-pill ${data.label}`;
    resultPill.textContent = data.label;

    // Confidence with colour band
    resultConf.innerHTML = `${confHtml(data.confidence)} confidence`;

    // Uncertainty badge (improved two-line version)
    uncertainBadge.style.display = data.uncertain ? 'inline-flex' : 'none';

    // Score bars
    scoreBars.innerHTML = buildScoreBars(data.scores);

    // Timestamp
    resultTs.textContent = `Scored at ${fmtTimestamp(new Date())}`;
  }
}

// ── Batch Page ────────────────────────────────────────────────────────────────

function initBatchPage() {
  const banner         = document.getElementById('model-banner');
  const errorBanner    = document.getElementById('error-banner');
  const batchInput     = document.getElementById('batch-input');
  const inputError     = document.getElementById('input-error');
  const batchBtn       = document.getElementById('batch-btn');
  const clearBtn       = document.getElementById('clear-btn');
  const reviewCount    = document.getElementById('review-count');
  const btnLabel       = document.getElementById('btn-label');
  const btnSpinner     = document.getElementById('btn-spinner');
  const fileInput      = document.getElementById('file-input');
  const uploadArea     = document.getElementById('upload-area');
  const fileLabel      = document.getElementById('file-label');
  const fileError      = document.getElementById('file-error');
  const batchProgress  = document.getElementById('batch-progress');
  const progressText   = document.getElementById('progress-text');
  const summarySection = document.getElementById('summary-section');
  const downloadBtn    = document.getElementById('download-csv-btn');

  let modelReady = false;
  let lastBatchResults = [];

  (async () => {
    modelReady = await checkHealth(banner);
    syncBatchButton();
  })();

  function getReviews() {
    return batchInput.value.split('\n').map(l => l.trim()).filter(Boolean);
  }

  function syncBatchButton() {
    const n = getReviews().length;
    reviewCount.textContent = n.toLocaleString();
    const tooMany = n > 2000;
    batchBtn.disabled = !modelReady || n === 0 || tooMany;
    if (tooMany) {
      showBanner(banner, 'warning', 'Maximum 2 000 reviews per batch. Remove some lines.');
    }
    // Hide field error once there's content
    if (n > 0) setFieldError(inputError, false);
  }

  batchInput.addEventListener('input', syncBatchButton);

  clearBtn.addEventListener('click', () => {
    batchInput.value = '';
    fileLabel.textContent = 'No file selected';
    summarySection.style.display = 'none';
    hideBanner(errorBanner);
    setFieldError(inputError, false);
    setFieldError(fileError, false);
    batchProgress.classList.remove('visible');
    lastBatchResults = [];
    syncBatchButton();
  });

  // Download results as CSV
  downloadBtn.addEventListener('click', () => {
    if (!lastBatchResults.length) return;

    const header = ['#', 'review', 'label', 'confidence', 'positive', 'neutral', 'negative', 'uncertain'];
    const rows = lastBatchResults.map((r, i) => [
      i + 1,
      r.text,
      r.label,
      r.confidence,
      r.scores.Positive,
      r.scores.Neutral,
      r.scores.Negative,
      r.uncertain,
    ]);

    const csvEscape = (val) => {
      const s = String(val).replace(/"/g, '""');
      return /[",\n]/.test(s) ? `"${s}"` : s;
    };

    const csv = [header, ...rows]
      .map(row => row.map(csvEscape).join(','))
      .join('\r\n');

    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    a.href = url;
    a.download = `sentiment-results-${timestamp}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // File upload
  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

  uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    if (!file) return;
    setFieldError(fileError, false);
    fileLabel.textContent = `Loading ${file.name}…`;

    const reader = new FileReader();
    reader.onload = e => {
      const lines = e.target.result
        .split('\n')
        .map(line => line.split(',')[0].replace(/^"|"$/g, '').trim())
        .filter(Boolean);

      // ── Validation: empty file
      if (lines.length === 0) {
        fileLabel.textContent = `${file.name} — no reviews found`;
        setFieldError(fileError, true);
        return;
      }

      batchInput.value = lines.join('\n');
      fileLabel.textContent = `${file.name} — ${lines.length.toLocaleString()} lines loaded`;
      setFieldError(fileError, false);
      syncBatchButton();
    };
    reader.onerror = () => {
      fileLabel.textContent = 'Failed to read file.';
      setFieldError(fileError, true);
    };
    reader.readAsText(file);
  }

  // Run analysis
  batchBtn.addEventListener('click', async () => {
    const reviews = getReviews();

    // ── Input validation
    if (reviews.length === 0) {
      setFieldError(inputError, true);
      batchInput.focus();
      return;
    }
    setFieldError(inputError, false);

    // ── Loading state
    setSpinner(batchBtn, btnLabel, btnSpinner, true, 'Run Analysis');
    batchProgress.classList.add('visible');
    progressText.textContent = `Scoring ${reviews.length.toLocaleString()} reviews…`;
    hideBanner(errorBanner);
    summarySection.style.display = 'none';

    try {
      const res = await fetch(API.predictBatch, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ reviews }),
      });

      if (!res.ok) {
        const err = await res.json();
        showBanner(errorBanner, 'error', `Error: ${err.detail ?? res.statusText}`);
        return;
      }

      renderBatchResults(await res.json());

    } catch {
      showBanner(errorBanner, 'error', 'Network error. Please try again.');
    } finally {
      batchProgress.classList.remove('visible');
      setSpinner(batchBtn, btnLabel, btnSpinner, false, 'Run Analysis');
      syncBatchButton();
    }
  });

  // ── Render helpers

  function renderBatchResults({ results, summary }) {
    // Stat cards
    document.getElementById('stat-grid').innerHTML = [
      statCard('Total',        summary.total,                          ''),
      statCard('Positive',     `${summary.positive_pct}%`,            'positive'),
      statCard('Neutral',      `${summary.neutral_pct}%`,             'neutral'),
      statCard('Negative',     `${summary.negative_pct}%`,            'negative'),
      statCard('Uncertain',    `${summary.uncertain_pct}%`,           'uncertain'),
      statCard('Avg Confidence', `${(summary.avg_confidence * 100).toFixed(1)}%`, ''),
    ].join('');

    // Highlight cards — positive, negative, most uncertain
    const hg = document.getElementById('highlight-grid');
    hg.innerHTML = '';
    if (summary.most_positive) {
      hg.innerHTML += highlightCard('Most Positive', summary.most_positive, 'var(--color-positive)');
    }
    if (summary.most_negative) {
      hg.innerHTML += highlightCard('Most Negative', summary.most_negative, 'var(--color-negative)');
    }
    if (summary.most_uncertain) {
      hg.innerHTML += highlightCard(
        'Most Uncertain',
        summary.most_uncertain,
        'var(--color-uncertain)',
        `Confidence: ${(summary.most_uncertain.confidence * 100).toFixed(1)}%`
      );
    }

    // Results table
    const tbody = document.getElementById('results-body');
    tbody.innerHTML = results.map((r, i) => `
      <tr>
        <td>${i + 1}</td>
        <td class="review-text">${escHtml(r.text.slice(0, 200))}${r.text.length > 200 ? '…' : ''}</td>
        <td>
          <span class="badge ${r.label}">${r.label}</span>
          ${r.uncertain ? '<span class="badge uncertain-dot" title="Low confidence — review manually">!</span>' : ''}
        </td>
        <td class="${confClass(r.confidence)}">${(r.confidence * 100).toFixed(1)}%</td>
        <td>${(r.scores.Positive * 100).toFixed(1)}%</td>
        <td>${(r.scores.Neutral  * 100).toFixed(1)}%</td>
        <td>${(r.scores.Negative * 100).toFixed(1)}%</td>
      </tr>
    `).join('');

    document.getElementById('results-count').textContent = `(${results.length.toLocaleString()})`;
    lastBatchResults = results;
    summarySection.style.display = 'block';
  }

  function statCard(label, value, cls) {
    return `
      <div class="stat-card ${cls}">
        <div class="stat-value">${value}</div>
        <div class="stat-label">${label}</div>
      </div>`;
  }

  function highlightCard(heading, result, accentColor, subline) {
    return `
      <div class="card" style="margin-bottom:0;border-top:3px solid ${accentColor};">
        <h2>${heading}</h2>
        ${subline ? `<div class="muted" style="margin:-0.5rem 0 0.6rem;font-size:0.8rem;">${escHtml(subline)}</div>` : ''}
        <p style="font-size:0.9rem;color:var(--color-text-muted);margin:0 0 0.75rem;">
          ${escHtml(result.text.slice(0, 200))}${result.text.length > 200 ? '…' : ''}
        </p>
        <div>${buildScoreBars(result.scores)}</div>
      </div>`;
  }
}

// ── Analytics Page ────────────────────────────────────────────────────────────

function initAnalyticsPage() {
  const emptyState = document.getElementById('empty-state');
  const content    = document.getElementById('analytics-content');

  fetch(API.analytics)
    .then(r => r.json())
    .then(data => {
      if (!data.history || data.history.length === 0) {
        emptyState.style.display = 'block';
        return;
      }
      content.style.display = 'block';
      renderLatest(data.latest);
      renderHistory(data.history);
      renderCharts(data.history);
      renderHighlightTexts(data.latest);
    })
    .catch(() => {
      emptyState.textContent = 'Failed to load analytics. Is the server running?';
      emptyState.style.display = 'block';
    });

  function renderLatest(run) {
    document.getElementById('latest-stats').innerHTML = `
      <div class="stat-card positive">
        <div class="stat-value">${run.positive_pct}%</div>
        <div class="stat-label">Positive (${run.positive_count})</div>
      </div>
      <div class="stat-card neutral">
        <div class="stat-value">${run.neutral_pct}%</div>
        <div class="stat-label">Neutral (${run.neutral_count})</div>
      </div>
      <div class="stat-card negative">
        <div class="stat-value">${run.negative_pct}%</div>
        <div class="stat-label">Negative (${run.negative_count})</div>
      </div>
      <div class="stat-card uncertain">
        <div class="stat-value">${run.uncertain_pct}%</div>
        <div class="stat-label">Uncertain (${run.uncertain_count})</div>
      </div>
      <div class="stat-card">
        <div class="stat-value ${confClass(run.avg_confidence)}">${(run.avg_confidence * 100).toFixed(1)}%</div>
        <div class="stat-label">Avg Confidence</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${run.total_reviews.toLocaleString()}</div>
        <div class="stat-label">Reviews scored</div>
      </div>
    `;

    const d = new Date(run.timestamp);
    document.getElementById('latest-meta').textContent =
      `Run ${run.run_id.slice(0, 8)}… · ${fmtTimestamp(d)}`;
  }

  function renderHistory(history) {
    document.getElementById('history-body').innerHTML = history.map(run => {
      const d = new Date(run.timestamp);
      return `
        <tr>
          <td title="${run.run_id}">${fmtTimestamp(d)}</td>
          <td>${run.total_reviews.toLocaleString()}</td>
          <td><span class="badge Positive">${run.positive_pct}%</span></td>
          <td><span class="badge Neutral">${run.neutral_pct}%</span></td>
          <td><span class="badge Negative">${run.negative_pct}%</span></td>
          <td>${run.uncertain_pct}%</td>
          <td class="${confClass(run.avg_confidence)}">${(run.avg_confidence * 100).toFixed(1)}%</td>
        </tr>`;
    }).join('');
  }

  function renderCharts(history) {
    const runs   = [...history].reverse(); // chronological
    const labels = runs.map(r => {
      const d = new Date(r.timestamp);
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    });

    const baseOpts = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { grid: { display: false } },
        y: { min: 0, max: 100, ticks: { callback: v => `${v}%` } },
      },
    };

    new Chart(document.getElementById('trend-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Positive', data: runs.map(r => r.positive_pct),
            borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.08)', tension: 0.3, fill: true },
          { label: 'Neutral',  data: runs.map(r => r.neutral_pct),
            borderColor: '#3498db', backgroundColor: 'rgba(52,152,219,0.08)', tension: 0.3, fill: true },
          { label: 'Negative', data: runs.map(r => r.negative_pct),
            borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.08)', tension: 0.3, fill: true },
        ],
      },
      options: baseOpts,
    });

    new Chart(document.getElementById('confidence-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Avg Confidence', data: runs.map(r => r.avg_confidence * 100),
            borderColor: '#2c3e50', backgroundColor: 'rgba(44,62,80,0.07)', tension: 0.3, fill: true },
          { label: 'Uncertain %',    data: runs.map(r => r.uncertain_pct),
            borderColor: '#f39c12', backgroundColor: 'rgba(243,156,18,0.07)', tension: 0.3, fill: true },
        ],
      },
      options: baseOpts,
    });
  }

  function renderHighlightTexts(run) {
    const el = document.getElementById('highlight-texts');
    let html = '';

    if (run.most_positive_text) {
      html += highlightTextCard('Most Positive', run.most_positive_text, 'var(--color-positive)');
    }
    if (run.most_negative_text) {
      html += highlightTextCard('Most Negative', run.most_negative_text, 'var(--color-negative)');
    }
    if (run.most_uncertain_text) {
      const confNote = run.most_uncertain_confidence != null
        ? ` · confidence ${(run.most_uncertain_confidence * 100).toFixed(1)}%`
        : '';
      html += highlightTextCard(
        `Most Uncertain${confNote}`,
        run.most_uncertain_text,
        'var(--color-uncertain)'
      );
    }

    el.innerHTML = html;
  }

  function highlightTextCard(heading, text, accentColor) {
    return `
      <div class="card" style="margin-bottom:0;border-top:3px solid ${accentColor};">
        <h2>${escHtml(heading)}</h2>
        <p style="font-size:0.9rem;color:var(--color-text-muted);margin:0;">${escHtml(text)}</p>
      </div>`;
  }
}

// ── Shared utility ────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}