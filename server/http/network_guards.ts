import net from 'node:net';

export function parseHttpUrl(raw: string) {
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function isLocalHost(hostname: string) {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1' || hostname === '0.0.0.0';
}

function normalizeIpLiteral(hostname: string) {
  const host = String(hostname || '').trim().toLowerCase();
  if (host.startsWith('::ffff:')) {
    return host.slice('::ffff:'.length);
  }
  return host;
}

function isPrivateIpv4(ip: string) {
  const parts = ip.split('.').map((part) => Number(part));
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }
  const [a, b] = parts;
  if (a === 10) return true; // 10.0.0.0/8
  if (a === 172 && b >= 16 && b <= 31) return true; // 172.16.0.0/12
  if (a === 192 && b === 168) return true; // 192.168.0.0/16
  if (a === 169 && b === 254) return true; // 169.254.0.0/16 link-local
  if (a === 100 && b >= 64 && b <= 127) return true; // 100.64.0.0/10 CGNAT
  return false;
}

function isPrivateIpv6(ip: string) {
  const value = String(ip || '').toLowerCase();
  if (value === '::1') return true; // loopback
  if (value.startsWith('fc') || value.startsWith('fd')) return true; // fc00::/7 ULA
  if (value.startsWith('fe8') || value.startsWith('fe9') || value.startsWith('fea') || value.startsWith('feb')) {
    return true; // fe80::/10 link-local
  }
  return false;
}

function isPrivateIpHost(hostname: string) {
  const host = normalizeIpLiteral(hostname);
  const ipVersion = net.isIP(host);
  if (ipVersion === 4) return isPrivateIpv4(host);
  if (ipVersion === 6) return isPrivateIpv6(host);
  return false;
}

function isLikelyLanHostname(hostname: string) {
  const host = String(hostname || '').trim().toLowerCase();
  if (!host) return false;
  if (host === 'localhost') return true;
  if (host.endsWith('.local') || host.endsWith('.lan') || host.endsWith('.home')) return true;
  if (!host.includes('.')) return true; // single-label hostname in LAN
  return false;
}

export function isAllowedTestUrl(parsed: URL) {
  const host = String(parsed.hostname || '').trim().toLowerCase();
  if (!host) return false;

  if (isLocalHost(host)) return true;

  const normalizedHost = normalizeIpLiteral(host);
  if (net.isIP(normalizedHost)) {
    // Allow only private LAN/local IP literals; block public IP literals.
    return isPrivateIpHost(normalizedHost);
  }

  if (parsed.protocol === 'http:') {
    // Plain HTTP is only for LAN-like hostnames.
    return isLikelyLanHostname(host);
  }

  // HTTPS domain names remain allowed (cloud providers).
  return true;
}

export function isBlockedPlaybackProxyUrl(parsed: URL) {
  const host = String(parsed.hostname || '').trim().toLowerCase();
  if (!host) return true;
  if (isLocalHost(host)) return true;
  if (isPrivateIpHost(host)) return true;
  if (isLikelyLanHostname(host)) return true;
  return false;
}

export function isRedirectStatus(status: number) {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308;
}

export function isMaskedKey(value: string) {
  const v = String(value || '').trim();
  if (!v) return false;
  if (v.includes('****') || v.includes('\u2022')) return true;
  if (v.includes('?\uFF34') || v.includes('\u0080')) return true; // legacy mojibake compatibility
  if (/^[A-Za-z0-9._-]{2,12}[*\u2022?]{3,}[A-Za-z0-9._-]{0,12}$/.test(v)) return true;
  if (/[^\x20-\x7E]/.test(v) && v.length <= 64) return true;
  return false;
}
