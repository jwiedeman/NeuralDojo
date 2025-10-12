export function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

export function formatCurrency(value) {
  if (!Number.isFinite(value)) return '—';
  return `$${Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  })}`;
}

export function formatPercent(value, digits = 2) {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatShares(value) {
  if (!Number.isFinite(value)) return '—';
  return `${Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 0
  })} sh`;
}

export function formatInteger(value) {
  if (!Number.isFinite(value)) return '—';
  return Number(value).toLocaleString();
}

export function formatSigned(value, digits = 3) {
  if (value == null || Number.isNaN(value)) return '—';
  const abs = Math.abs(value).toFixed(digits);
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  return `${sign}${abs}`;
}

export function prettifyReason(reason) {
  if (!reason) return null;
  const normalized = String(reason).replace(/_/g, ' ').trim();
  if (!normalized) return null;
  return normalized.replace(/\b\w/g, char => char.toUpperCase());
}
