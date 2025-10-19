const playerInput = document.getElementById('player');
const playerList = document.getElementById('player-list');
const positionInput = document.getElementById('position');
const recommendBtn = document.getElementById('recommend-btn');
const errorBox = document.getElementById('error');
const table = document.getElementById('results-table');
const tbody = table.querySelector('tbody');

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
