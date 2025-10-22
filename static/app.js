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
}

playerInput.addEventListener('input', (e) => {
  const q = e.target.value.trim();
  if (q.length >= 2) {
    fetchPlayers(q);
  }
});

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
