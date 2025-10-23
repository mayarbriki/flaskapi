const playerInput = document.getElementById('player');
const playerList = document.getElementById('player-list');
const positionInput = document.getElementById('position');
const recommendBtn = document.getElementById('recommend-btn');
const errorBox = document.getElementById('error');
const table = document.getElementById('results-table');
const tbody = table.querySelector('tbody');
const predictBtn = document.getElementById('predict-btn');
const predictForm = document.getElementById('predict-form');
const predictErr = document.getElementById('predict-error');
const predictOut = document.getElementById('predict-output');
const ageRange = document.getElementById('pf-age');
const ageVal = document.getElementById('age-val');
const hRange = document.getElementById('pf-height');
const hVal = document.getElementById('h-val');
const wRange = document.getElementById('pf-weight');
const wVal = document.getElementById('w-val');
const selPos = document.getElementById('pf-positions');
const selFoot = document.getElementById('pf-foot');
const selBody = document.getElementById('pf-body');
const inpNat = document.getElementById('pf-nationality');
const natList = document.getElementById('nationality-list');

// AI Scout Report elements
const scoutForm = document.getElementById('scout-form');
const scoutBtn = document.getElementById('scout-btn');
const scoutInput = document.getElementById('sr-player');
const scoutErr = document.getElementById('scout-error');
const scoutOut = document.getElementById('scout-output');
const scoutLang = document.getElementById('sr-lang');
const translateBtn = document.getElementById('translate-btn');
const scoutTransErr = document.getElementById('scout-trans-error');
const scoutTranslation = document.getElementById('scout-translation');
const scoutProvider = document.getElementById('scout-provider');
// Browse Players elements
const bpForm = document.getElementById('browse-form');
const bpQ = document.getElementById('bp-q');
const bpNationality = document.getElementById('bp-nationality');
const bpPosition = document.getElementById('bp-position');
const bpMinOverall = document.getElementById('bp-min-overall');
const bpMaxOverall = document.getElementById('bp-max-overall');
const bpMinAge = document.getElementById('bp-min-age');
const bpMaxAge = document.getElementById('bp-max-age');
const bpMinValue = document.getElementById('bp-min-value');
const bpMaxValue = document.getElementById('bp-max-value');
const bpSort = document.getElementById('bp-sort');
const bpPageSize = document.getElementById('bp-page-size');
const bpApply = document.getElementById('bp-apply');
const bpErr = document.getElementById('bp-error');
const bpTable = document.getElementById('bp-table');
const bpTbody = bpTable ? bpTable.querySelector('tbody') : null;
const bpPager = document.getElementById('bp-pager');
const bpPrev = document.getElementById('bp-prev');
const bpNext = document.getElementById('bp-next');
const bpInfo = document.getElementById('bp-info');
const bpExport = document.getElementById('bp-export');
const bpSemantic = document.getElementById('bp-semantic');
// Player detail modal
const pdModal = document.getElementById('player-detail');
const pdTitle = document.getElementById('pd-title');
const pdBody = document.getElementById('pd-body');
const pdClose = document.getElementById('pd-close');
// Top stats elements
const tsForm = document.getElementById('top-form');
const tsStat = document.getElementById('ts-stat');
const tsN = document.getElementById('ts-n');
const tsPos = document.getElementById('ts-position');
const tsNat = document.getElementById('ts-nationality');
const tsApply = document.getElementById('ts-apply');
const tsErr = document.getElementById('ts-error');
const tsTable = document.getElementById('ts-table');
const tsTbody = tsTable ? tsTable.querySelector('tbody') : null;

let suggestAbort = null;

async function fetchPlayers(query) {
  try {
    if (suggestAbort) suggestAbort.abort();
    suggestAbort = new AbortController();
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    params.set('limit', '100');
    const res = await fetch(`/players?${params.toString()}`, { signal: suggestAbort.signal });
    if (!res.ok) throw new Error('Failed to load players');
    const names = await res.json();
    playerList.innerHTML = '';
    names.forEach(n => {
      const opt = document.createElement('option');
      opt.value = n;
      playerList.appendChild(opt);
    });
  } catch (e) {
    // Ignore abort errors
  }

// ----- Dataset Q&A -----
const qaAsk = document.getElementById('qa-ask');
const qaQ = document.getElementById('qa-q');
const qaErr = document.getElementById('qa-error');
const qaOut = document.getElementById('qa-out');
if (qaAsk) qaAsk.addEventListener('click', async () => {
  if (qaErr) qaErr.hidden = true;
  if (qaOut) { qaOut.hidden = true; qaOut.textContent = ''; }
  const q = (qaQ?.value || '').trim();
  if (!q) { if (qaErr) { qaErr.textContent = 'Enter a question.'; qaErr.hidden = false; } return; }
  try {
    const res = await fetch('/api/qa', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'QA failed');
    const items = Array.isArray(data.items) ? data.items : [];
    const summary = data.summary || '';
    if (items.length) {
      // Build a small table
      const cols = Object.keys(items[0]);
      let html = '<table class="compact"><thead><tr>' + cols.map(c => `<th>${escapeHtml(String(c))}</th>`).join('') + '</tr></thead><tbody>';
      items.forEach(row => {
        html += '<tr>' + cols.map(c => `<td>${escapeHtml(String(row[c] ?? ''))}</td>`).join('') + '</tr>';
      });
      html += '</tbody></table>';
      if (summary) html += `\n\n<div class="muted" style="margin-top:.5rem;">${escapeHtml(summary)}</div>`;
      qaOut.innerHTML = html;
      qaOut.hidden = false;
    } else if (summary) {
      qaOut.textContent = summary;
      qaOut.hidden = false;
    } else {
      qaOut.textContent = 'No results.';
      qaOut.hidden = false;
    }
  } catch (e) {
    if (qaErr) { qaErr.textContent = e.message; qaErr.hidden = false; }
  }
});

function showPlayerDetail(name, record) {
  if (!pdModal || !pdBody || !pdTitle) return;
  pdTitle.textContent = name || 'Player Detail';
  const entries = Object.entries(record || {});
  const html = entries
    .map(([k, v]) => `<div class="row"><strong>${escapeHtml(String(k))}:</strong> <span>${escapeHtml(String(v ?? ''))}</span></div>`)
    .join('');
  pdBody.innerHTML = html || '<em>No data</em>';
  pdModal.hidden = false;
  document.body.classList.add('modal-open');
}

function closePlayerDetail() {
  if (!pdModal) return;
  pdModal.hidden = true;
  document.body.classList.remove('modal-open');
}

if (pdClose) pdClose.addEventListener('click', closePlayerDetail);
if (pdModal) pdModal.addEventListener('click', (e) => { if (e.target === pdModal) closePlayerDetail(); });
// Position Classify in modal
const pdClassify = document.getElementById('pd-classify');
const pdClassifyErr = document.getElementById('pd-classify-error');
if (pdClassify) pdClassify.addEventListener('click', async () => {
  const name = pdTitle?.textContent || '';
  if (!name) return;
  if (pdClassifyErr) pdClassifyErr.hidden = true;
  try {
    const res = await fetch('/api/position-classify', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Classify failed');
    alert(`Predicted role: ${JSON.stringify(data)}`);
  } catch (e) {
    if (pdClassifyErr) { pdClassifyErr.textContent = e.message; pdClassifyErr.hidden = false; }
  }
});

// ----- Top Players wiring -----
async function loadTopStats() {
  if (tsErr) tsErr.hidden = true;
  if (tsTable) tsTable.hidden = true;
  if (tsTbody) tsTbody.innerHTML = '';
  const params = new URLSearchParams();
  if (tsStat?.value) params.set('stat', tsStat.value);
  const n = tsN?.value ? parseInt(tsN.value, 10) : 10;
  params.set('n', String(isFinite(n) ? n : 10));
  if (tsPos?.value) params.set('position', tsPos.value.trim());
  if (tsNat?.value) params.set('nationality', tsNat.value.trim());
  try {
    const res = await fetch(`/api/stats/top?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Failed to load top stats');
    const items = Array.isArray(data.items) ? data.items : [];
    items.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><button class="link" data-pname="${escapeHtml(String(row.name || ''))}">${escapeHtml(String(row.name || ''))}</button></td>
        <td>${escapeHtml(String(row.nationality || ''))}</td>
        <td>${escapeHtml(String(row.positions || ''))}</td>
        <td>${escapeHtml(String(row.overall_rating ?? ''))}</td>
        <td>${escapeHtml(String(row.value_euro ?? ''))}</td>
        <td>${escapeHtml(String(row.stat_value ?? ''))}</td>
      `;
      tsTbody?.appendChild(tr);
    });
    if (tsTable) tsTable.hidden = items.length === 0;
    // attach detail click handlers
    if (tsTbody) {
      tsTbody.querySelectorAll('button.link[data-pname]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const pname = e.currentTarget.getAttribute('data-pname') || '';
          if (!pname) return;
          try {
            const res = await fetch(`/api/players/${encodeURIComponent(pname)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || 'Failed to load player');
            showPlayerDetail(pname, data);
          } catch (err) {
            alert(err.message || 'Failed to load player');
          }
        });
      });
    }
  } catch (e) {
    if (tsErr) { tsErr.textContent = e.message; tsErr.hidden = false; }
  }
}

if (tsApply) tsApply.addEventListener('click', () => loadTopStats());
// initial load for default metric
if (tsForm) loadTopStats();
}

// ----- Semantic Search -----
if (bpSemantic) bpSemantic.addEventListener('click', async () => {
  const q = (bpQ?.value || '').trim();
  if (!q) { alert('Enter a search query first.'); return; }
  try {
    const res = await fetch(`/api/search/semantic?q=${encodeURIComponent(q)}&k=${encodeURIComponent(bpPageSize?.value || '20')}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Semantic search failed');
    if (bpErr) bpErr.hidden = true;
    if (bpTbody) bpTbody.innerHTML = '';
    const items = Array.isArray(data.items) ? data.items : [];
    items.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><button class="link" data-pname="${escapeHtml(String(row.name || ''))}">${escapeHtml(String(row.name || ''))}</button></td>
        <td>${escapeHtml(String(row.nationality || ''))}</td>
        <td>${escapeHtml(String(row.overall_rating ?? ''))}</td>
        <td>${escapeHtml(String(row.age ?? ''))}</td>
        <td>${escapeHtml(String(row.value_euro ?? ''))}</td>
        <td>${escapeHtml(String(row.positions || ''))}</td>
      `;
      bpTbody?.appendChild(tr);
    });
    if (bpTable) bpTable.hidden = items.length === 0;
    if (bpPager) bpPager.hidden = true;
    if (bpTbody) {
      bpTbody.querySelectorAll('button.link[data-pname]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const pname = e.currentTarget.getAttribute('data-pname') || '';
          if (!pname) return;
          try {
            const res = await fetch(`/api/players/${encodeURIComponent(pname)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || 'Failed to load player');
            showPlayerDetail(pname, data);
          } catch (err) {
            alert(err.message || 'Failed to load player');
          }
        });
      });
    }
  } catch (e) {
    if (bpErr) { bpErr.textContent = e.message; bpErr.hidden = false; }
  }
});

playerInput.addEventListener('input', (e) => {
  const q = e.target.value.trim();
  if (q.length >= 2) {
    fetchPlayers(q);
  }
});

// Mirror suggestions on the scout input as well
if (scoutInput) {
  scoutInput.addEventListener('input', (e) => {
    const q = e.target.value.trim();
    if (q.length >= 2) {
      fetchPlayers(q);
    }
  });
}

if (translateBtn) {
  translateBtn.addEventListener('click', async () => {
    if (scoutTransErr) scoutTransErr.hidden = true;
    if (scoutTranslation) { scoutTranslation.hidden = true; scoutTranslation.textContent = ''; }

    const reportText = (scoutOut && !scoutOut.hidden && typeof scoutOut.textContent === 'string') ? scoutOut.textContent.trim() : '';
    if (!reportText) {
      if (scoutTransErr) { scoutTransErr.textContent = 'Generate a report first.'; scoutTransErr.hidden = false; }
      return;
    }
    const targetLang = (scoutLang?.value || '').trim() || 'English';

    try {
      translateBtn.disabled = true;
      if (scoutForm) scoutForm.classList.add('loading');
      const res = await fetch('/api/translate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: reportText, target_lang: targetLang })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.error || 'Translation failed');
      }
      if (scoutTranslation) {
        scoutTranslation.textContent = data.translated_text || '';
        scoutTranslation.hidden = false;
        scoutTranslation.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      if (scoutProvider) {
        const pv = data.provider || 'unknown';
        scoutProvider.textContent = `Provider: ${pv}`;
        scoutProvider.hidden = false;
      }
    } catch (e) {
      if (scoutTransErr) { scoutTransErr.textContent = e.message; scoutTransErr.hidden = false; }
    } finally {
      translateBtn.disabled = false;
      if (scoutForm) scoutForm.classList.remove('loading');
    }
  });
}

recommendBtn.addEventListener('click', async () => {
  const player = playerInput.value.trim();
  const position = positionInput.value.trim();
  errorBox.hidden = true;
  table.hidden = true;
  tbody.innerHTML = '';

  if (!player) {
    showError('Please enter a player name.');
    return;
  }

  try {
    recommendBtn.disabled = true;
    const params = new URLSearchParams({ player });
    if (position) params.set('position', position);

    const res = await fetch(`/recommend?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error || 'Request failed');
    }

    if (!Array.isArray(data) || data.length === 0) {
      showError('No recommendations found.');
      return;
    }

    data.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${escapeHtml(row.name)}</td>
        <td>${escapeHtml(String(row.nationality))}</td>
        <td>${escapeHtml(String(row.overall_rating))}</td>
        <td>${escapeHtml(String(row.value_euro))}</td>
        <td>${escapeHtml(String(row.positions_original))}</td>
      `;
      tbody.appendChild(tr);
    });
    table.hidden = false;
  } catch (e) {
    showError(e.message);
  } finally {
    recommendBtn.disabled = false;
  }
});

// Generate AI Scout Report handler
if (scoutBtn) {
  scoutBtn.addEventListener('click', async () => {
    const playerName = (scoutInput?.value || '').trim() || (playerInput?.value || '').trim();
    if (scoutErr) scoutErr.hidden = true;
    if (scoutOut) { scoutOut.hidden = true; scoutOut.textContent = ''; }
    if (scoutTransErr) scoutTransErr.hidden = true;
    if (scoutTranslation) { scoutTranslation.hidden = true; scoutTranslation.textContent = ''; }
    if (scoutProvider) { scoutProvider.hidden = true; scoutProvider.textContent = ''; }

    if (!playerName) {
      if (scoutErr) { scoutErr.textContent = 'Please enter a player name.'; scoutErr.hidden = false; }
      return;
    }

    try {
      scoutBtn.disabled = true;
      if (scoutForm) scoutForm.classList.add('loading');
      const res = await fetch('/api/scout-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_name: playerName })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.error || 'Request failed');
      }
      if (scoutOut) {
        scoutOut.textContent = data.report || '';
        scoutOut.hidden = false;
        scoutOut.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    } catch (e) {
      if (scoutErr) { scoutErr.textContent = e.message; scoutErr.hidden = false; }
    } finally {
      scoutBtn.disabled = false;
      if (scoutForm) scoutForm.classList.remove('loading');
    }
  });
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function escapeHtml(unsafe) {
  return unsafe
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

// Preload some player names on first load
fetchPlayers('');

// Metadata loader
async function loadMetadata() {
  try {
    const res = await fetch('/metadata');
    if (!res.ok) return;
    const meta = await res.json();

    // Positions
    selPos.innerHTML = '';
    (meta.positions || []).forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      selPos.appendChild(opt);
    });

    // Preferred foot
    selFoot.innerHTML = '';
    (meta.preferred_foot || ['Right','Left']).forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      selFoot.appendChild(opt);
    });

    // Body type
    selBody.innerHTML = '';
    (meta.body_type || ['Normal']).forEach(b => {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = b;
      selBody.appendChild(opt);
    });

    // Nationality datalist
    natList.innerHTML = '';
    (meta.nationality_top || []).forEach(n => {
      const opt = document.createElement('option');
      opt.value = n;
      natList.appendChild(opt);
    });
  } catch {}
}

loadMetadata();

// Range outputs
function bindRange(rangeEl, outEl) {
  if (!rangeEl || !outEl) return;
  const update = () => { outEl.textContent = rangeEl.value; };
  rangeEl.addEventListener('input', update);
  update();
}
bindRange(ageRange, ageVal);
bindRange(hRange, hVal);
bindRange(wRange, wVal);
document.querySelectorAll('input[type="range"][name]').forEach(r => {
  const out = document.querySelector(`[data-out="${r.name}"]`);
  bindRange(r, out);
});

// Predict Overall handler (form submit)
if (predictForm) {
  predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    predictErr.hidden = true;
    predictOut.hidden = true;
    try {
      predictBtn.disabled = true;
      predictForm.classList.add('loading');
      // Collect form values
      const form = new FormData(predictForm);
      const payload = {};
      form.forEach((v, k) => {
        if (typeof v === 'string' && v.trim() !== '' && !isNaN(Number(v)) && !['preferred_foot','body_type','nationality','positions'].includes(k)) {
          payload[k] = Number(v);
        } else {
          payload[k] = v;
        }
      });

      const res = await fetch('/predict/overall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.error || 'Prediction failed');
      }
      predictOut.innerHTML = `
        <div>Predicted Overall: <span class="badge">${data.overall_rounded}</span>
          <small style="opacity:.8">(raw: ${Number(data.overall).toFixed(2)})</small>
        </div>`;
      predictOut.hidden = false;
      predictOut.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (e) {
      predictErr.textContent = e.message;
      predictErr.hidden = false;
    } finally {
      predictBtn.disabled = false;
      predictForm.classList.remove('loading');
    }
  });
}

// Sidebar functionality
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
let sidebarOverlay = null;

// Create overlay for mobile
function createOverlay() {
  if (!sidebarOverlay) {
    sidebarOverlay = document.createElement('div');
    sidebarOverlay.className = 'sidebar-overlay';
    document.body.appendChild(sidebarOverlay);
    
    sidebarOverlay.addEventListener('click', closeSidebar);
  }
}

function toggleSidebar() {
  sidebar.classList.toggle('active');
  sidebarToggle.classList.toggle('active');
  
  if (window.innerWidth <= 767) {
    createOverlay();
    sidebarOverlay.classList.toggle('active');
  }
}

function closeSidebar() {
  sidebar.classList.remove('active');
  sidebarToggle.classList.remove('active');
  if (sidebarOverlay) {
    sidebarOverlay.classList.remove('active');
  }
}

function scrollToSection(elementId) {
  const element = document.getElementById(elementId);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    closeSidebar();
  }
}

function showAbout() {
  alert('FIFA Player Recommender\n\nFind similar players and predict overall ratings using machine learning.\n\nFeatures:\n• Player recommendations\n• Overall rating prediction\n• Advanced player statistics');
  closeSidebar();
}

// Event listeners
sidebarToggle.addEventListener('click', toggleSidebar);

// Close sidebar when clicking outside on desktop
document.addEventListener('click', (e) => {
  if (window.innerWidth > 767 && 
      !sidebar.contains(e.target) && 
      !sidebarToggle.contains(e.target) &&
      sidebar.classList.contains('active')) {
    closeSidebar();
  }
});

// Handle window resize
window.addEventListener('resize', () => {
  if (window.innerWidth > 767 && sidebarOverlay) {
    sidebarOverlay.classList.remove('active');
  }
});

// ----- Browse Players wiring -----
let bpPage = 1;
async function loadBrowse(forcePage) {
  if (forcePage) bpPage = forcePage;
  if (bpErr) bpErr.hidden = true;
  if (bpTable) bpTable.hidden = true;
  if (bpTbody) bpTbody.innerHTML = '';

  const params = new URLSearchParams();
  if (bpQ?.value) params.set('q', bpQ.value.trim());
  if (bpNationality?.value) params.set('nationality', bpNationality.value.trim());
  if (bpPosition?.value) params.set('position', bpPosition.value.trim());
  if (bpMinOverall?.value) params.set('min_overall', bpMinOverall.value.trim());
  if (bpMaxOverall?.value) params.set('max_overall', bpMaxOverall.value.trim());
  if (bpMinAge?.value) params.set('min_age', bpMinAge.value.trim());
  if (bpMaxAge?.value) params.set('max_age', bpMaxAge.value.trim());
  if (bpMinValue?.value) params.set('min_value', bpMinValue.value.trim());
  if (bpMaxValue?.value) params.set('max_value', bpMaxValue.value.trim());
  if (bpSort?.value) params.set('sort', bpSort.value);
  const ps = bpPageSize?.value ? parseInt(bpPageSize.value, 10) : 20;
  params.set('page', String(bpPage));
  params.set('page_size', String(isFinite(ps) ? ps : 20));

  try {
    const res = await fetch(`/api/players?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Failed to load players');
    const items = Array.isArray(data.items) ? data.items : [];
    items.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><button class="link" data-pname="${escapeHtml(String(row.name || ''))}">${escapeHtml(String(row.name || ''))}</button></td>
        <td>${escapeHtml(String(row.nationality || ''))}</td>
        <td>${escapeHtml(String(row.overall_rating ?? ''))}</td>
        <td>${escapeHtml(String(row.age ?? ''))}</td>
        <td>${escapeHtml(String(row.value_euro ?? ''))}</td>
        <td>${escapeHtml(String(row.positions_original || ''))}</td>
      `;
      bpTbody?.appendChild(tr);
    });
    // attach click handlers for details
    if (bpTbody) {
      bpTbody.querySelectorAll('button.link[data-pname]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const pname = e.currentTarget.getAttribute('data-pname') || '';
          if (!pname) return;
          try {
            const res = await fetch(`/api/players/${encodeURIComponent(pname)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || 'Failed to load player');
            showPlayerDetail(pname, data);
          } catch (err) {
            alert(err.message || 'Failed to load player');
          }
        });
      });
    }
    if (bpTable) bpTable.hidden = items.length === 0;
    if (bpPager && bpInfo) {
      const total = Number(data.total || 0);
      const page = Number(data.page || 1);
      const totalPages = Number(data.total_pages || 1);
      bpInfo.textContent = `Page ${page} / ${totalPages} — ${total} players`;
      bpPager.hidden = totalPages <= 1 && items.length === 0;
      bpPrev.disabled = page <= 1;
      bpNext.disabled = page >= totalPages;
    }
  } catch (e) {
    if (bpErr) { bpErr.textContent = e.message; bpErr.hidden = false; }
  }
}

function resetAndLoadBrowse() {
  bpPage = 1;
  loadBrowse(1);
}

// ----- Export CSV for Browse filters -----
if (bpExport) bpExport.addEventListener('click', async () => {
  const payload = {
    q: bpQ?.value?.trim() || '',
    nationality: bpNationality?.value?.trim() || '',
    position: bpPosition?.value?.trim() || '',
    min_overall: bpMinOverall?.value || '',
    max_overall: bpMaxOverall?.value || '',
    min_age: bpMinAge?.value || '',
    max_age: bpMaxAge?.value || '',
    min_value: bpMinValue?.value || '',
    max_value: bpMaxValue?.value || ''
  };
  try {
    const res = await fetch('/api/players/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || 'Export failed');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'players_export.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert(e.message || 'Export failed');
  }
});

// ----- Compare -----
const cpForm = document.getElementById('compare-form');
const cpPlayers = document.getElementById('cp-players');
const cpMetrics = document.getElementById('cp-metrics');
const cpApply = document.getElementById('cp-apply');
const cpErr = document.getElementById('cp-error');
const cpTable = document.getElementById('cp-table');
const cpHeadRow = document.getElementById('cp-head-row');
const cpTbody = cpTable ? cpTable.querySelector('tbody') : null;

async function loadCompare() {
  if (cpErr) cpErr.hidden = true;
  if (cpTable) cpTable.hidden = true;
  if (cpTbody) cpTbody.innerHTML = '';
  if (cpHeadRow) cpHeadRow.innerHTML = '<th>Name</th>';
  const params = new URLSearchParams();
  if (cpPlayers?.value) params.set('players', cpPlayers.value);
  if (cpMetrics?.value) params.set('metrics', cpMetrics.value);
  try {
    const res = await fetch(`/api/compare?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Compare failed');
    const metrics = Array.isArray(data.metrics) ? data.metrics : [];
    metrics.forEach(m => {
      const th = document.createElement('th'); th.textContent = m; cpHeadRow?.appendChild(th);
    });
    (data.items || []).forEach(row => {
      const tr = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.innerHTML = `<button class="link" data-pname="${escapeHtml(String(row.name || ''))}">${escapeHtml(String(row.name || ''))}</button>`;
      tr.appendChild(nameCell);
      metrics.forEach(m => {
        const td = document.createElement('td');
        td.textContent = row[m] == null ? '' : String(row[m]);
        tr.appendChild(td);
      });
      cpTbody?.appendChild(tr);
    });
    if (cpTbody) {
      cpTbody.querySelectorAll('button.link[data-pname]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const pname = e.currentTarget.getAttribute('data-pname') || '';
          if (!pname) return;
          try {
            const res = await fetch(`/api/players/${encodeURIComponent(pname)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || 'Failed to load player');
            showPlayerDetail(pname, data);
          } catch (err) {
            alert(err.message || 'Failed to load player');
          }
        });
      });
    }
    if (cpTable) cpTable.hidden = false;
  } catch (e) {
    if (cpErr) { cpErr.textContent = e.message; cpErr.hidden = false; }
  }
}
if (cpApply) cpApply.addEventListener('click', () => loadCompare());

// Explain Comparison (LLM)
const cpExplain = document.getElementById('cp-explain');
const cpExplainErr = document.getElementById('cp-explain-error');
const cpExplainOut = document.getElementById('cp-explain-out');
if (cpExplain) cpExplain.addEventListener('click', async () => {
  if (cpExplainErr) cpExplainErr.hidden = true;
  if (cpExplainOut) { cpExplainOut.hidden = true; cpExplainOut.textContent = ''; }
  const players = (cpPlayers?.value || '').split(',').map(s => s.trim()).filter(Boolean);
  const metrics = (cpMetrics?.value || '').split(',').map(s => s.trim()).filter(Boolean);
  if (!players.length) { if (cpExplainErr) { cpExplainErr.textContent = 'Enter at least one player.'; cpExplainErr.hidden = false; } return; }
  try {
    const res = await fetch('/api/compare/explain', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ players, metrics })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Explain failed');
    if (cpExplainOut) { cpExplainOut.textContent = data.analysis || ''; cpExplainOut.hidden = false; }
  } catch (e) {
    if (cpExplainErr) { cpExplainErr.textContent = e.message; cpExplainErr.hidden = false; }
  }
});

// ----- Anomalies -----
const anForm = document.getElementById('an-form');
const anN = document.getElementById('an-n');
const anPos = document.getElementById('an-position');
const anNat = document.getElementById('an-nationality');
const anApply = document.getElementById('an-apply');
const anErr = document.getElementById('an-error');
const anTable = document.getElementById('an-table');
const anTbody = anTable ? anTable.querySelector('tbody') : null;

async function loadAnomalies() {
  if (anErr) anErr.hidden = true;
  if (anTable) anTable.hidden = true;
  if (anTbody) anTbody.innerHTML = '';
  try {
    const body = {
      n: anN?.value || 20,
      position: anPos?.value || '',
      nationality: anNat?.value || ''
    };
    const res = await fetch('/api/anomalies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Anomaly detection failed');
    (data.items || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><button class=\"link\" data-pname=\"${escapeHtml(String(row.name || ''))}\">${escapeHtml(String(row.name || ''))}</button></td>
        <td>${escapeHtml(String(row.nationality || ''))}</td>
        <td>${escapeHtml(String(row.positions || ''))}</td>
        <td>${escapeHtml(String(row.overall_rating ?? ''))}</td>
        <td>${escapeHtml(String(row.value_euro ?? ''))}</td>
        <td>${escapeHtml(String(row.age ?? ''))}</td>
        <td>${escapeHtml(String(row.anomaly_score ?? ''))}</td>
      `;
      anTbody?.appendChild(tr);
    });
    if (anTable) anTable.hidden = false;
    if (anTbody) {
      anTbody.querySelectorAll('button.link[data-pname]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const pname = e.currentTarget.getAttribute('data-pname') || '';
          if (!pname) return;
          try {
            const res = await fetch(`/api/players/${encodeURIComponent(pname)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || 'Failed to load player');
            showPlayerDetail(pname, data);
          } catch (err) {
            alert(err.message || 'Failed to load player');
          }
        });
      });
    }
  } catch (e) {
    if (anErr) { anErr.textContent = e.message; anErr.hidden = false; }
  }
}
if (anApply) anApply.addEventListener('click', () => loadAnomalies());

// ----- Histogram -----
const hgForm = document.getElementById('hg-form');
const hgStat = document.getElementById('hg-stat');
const hgBins = document.getElementById('hg-bins');
const hgPos = document.getElementById('hg-position');
const hgNat = document.getElementById('hg-nationality');
const hgApply = document.getElementById('hg-apply');
const hgErr = document.getElementById('hg-error');
const hgOut = document.getElementById('hg-output');
const hgCanvas = document.getElementById('hg-canvas');

async function loadHistogram() {
  if (hgErr) hgErr.hidden = true;
  if (hgOut) { hgOut.hidden = true; hgOut.textContent = ''; }
  // clear canvas
  if (hgCanvas) {
    const ctx = hgCanvas.getContext('2d');
    ctx.clearRect(0, 0, hgCanvas.width, hgCanvas.height);
  }
  const params = new URLSearchParams();
  if (hgStat?.value) params.set('stat', hgStat.value);
  if (hgBins?.value) params.set('bins', hgBins.value);
  if (hgPos?.value) params.set('position', hgPos.value.trim());
  if (hgNat?.value) params.set('nationality', hgNat.value.trim());
  try {
    const res = await fetch(`/api/stats/histogram?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Failed to load histogram');
    // draw chart
    const counts = Array.isArray(data.counts) ? data.counts : [];
    const bins = Array.isArray(data.bins) ? data.bins : [];
    if (hgCanvas && counts.length > 0 && bins.length >= counts.length + 1) {
      drawHistogram(hgCanvas, counts, bins);
    }
    // keep raw output collapsed
    hgOut.textContent = JSON.stringify({ stat: data.stat, counts: data.counts?.length, bins: data.bins?.length }, null, 2);
    hgOut.hidden = true;
  } catch (e) {
    if (hgErr) { hgErr.textContent = e.message; hgErr.hidden = false; }
  }
}
if (hgApply) hgApply.addEventListener('click', () => loadHistogram());

function drawHistogram(canvas, counts, bins) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const margin = { left: 50, right: 20, top: 20, bottom: 40 };
  const w = W - margin.left - margin.right;
  const h = H - margin.top - margin.bottom;

  const maxCount = Math.max(1, ...counts);
  const n = counts.length;
  const barW = w / n;

  // axes
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + h);
  ctx.lineTo(margin.left + w, margin.top + h);
  ctx.stroke();

  // bars
  ctx.fillStyle = '#3b82f6';
  for (let i = 0; i < n; i++) {
    const val = counts[i];
    const bh = (val / maxCount) * h;
    const x = margin.left + i * barW + 1;
    const y = margin.top + (h - bh);
    ctx.fillRect(x, y, Math.max(1, barW - 2), bh);
  }

  // y-axis ticks
  ctx.fillStyle = '#444';
  ctx.font = '12px sans-serif';
  const ticks = 4;
  for (let t = 0; t <= ticks; t++) {
    const v = Math.round((t / ticks) * maxCount);
    const y = margin.top + h - (t / ticks) * h;
    ctx.fillText(String(v), 5, y + 4);
    ctx.strokeStyle = '#eee';
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + w, y);
    ctx.stroke();
  }

  // x-axis labels (bin edges, show ~6 labels max)
  const labelCount = Math.min(6, bins.length);
  const step = Math.max(1, Math.floor((bins.length - 1) / (labelCount - 1)));
  for (let i = 0; i < bins.length; i += step) {
    const frac = i / (bins.length - 1);
    const x = margin.left + frac * w;
    const label = String(Math.round(bins[i]));
    ctx.fillText(label, x - 8, margin.top + h + 16);
  }
}

if (bpApply) bpApply.addEventListener('click', () => resetAndLoadBrowse());
if (bpPrev) bpPrev.addEventListener('click', () => { if (bpPage > 1) loadBrowse(bpPage - 1); });
if (bpNext) bpNext.addEventListener('click', () => loadBrowse(bpPage + 1));

// Auto-load initial browse view on first paint
if (bpForm) {
  // Small debounce on typing in text inputs
  const debounced = (fn, ms=300) => { let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; };
  [bpQ].forEach(el => el && el.addEventListener('input', debounced(() => resetAndLoadBrowse(), 400)));
  loadBrowse(1);
}
