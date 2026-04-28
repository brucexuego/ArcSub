import os from 'node:os';
import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { PathManager } from '../path_manager.js';

const execFileAsync = promisify(execFile);

type AcceleratorKind = 'gpu' | 'npu';

export interface AcceleratorSnapshot {
  id: string;
  kind: AcceleratorKind;
  vendor: 'intel';
  model: string;
  utilization: number;
  vramUsedGB?: number;
  vramTotalGB?: number;
  luid?: string;
  memorySource?: 'dedicated' | 'shared';
  engineTypes?: string[];
  physIndex?: number;
  taskManagerIndex?: number;
}

interface CpuTicksSnapshot {
  idle: number;
  total: number;
}

interface WindowsIntelDxdiagEntry {
  model: string;
  dedicatedGB?: number;
  totalGB?: number;
}

interface WindowsIntelDevices {
  gpus: Array<{
    model: string;
    dedicatedGB?: number;
    taskManagerOrder?: number;
  }>;
  npuModels: string[];
}

interface WindowsGpuMetric {
  luid: string;
  physIndex: number;
  utilization: number;
  utilization3d: number;
  utilizationCompute: number;
  engineTypes: string[];
  dedicatedUsedGB?: number;
  sharedUsedGB?: number;
}

export interface SystemResourceSnapshot {
  cpu: number;
  ram: number;
  gpu: number;
  vram: number;
  ramUsedGB: number;
  ramTotalGB: number;
  vramUsedGB?: number;
  vramTotalGB?: number;
  platform: string;
  cpuModel: string;
  gpuModel?: string;
  accelerators: AcceleratorSnapshot[];
}

const WINDOWS_DXDIAG_CACHE_MS = 10 * 60 * 1000;
const WINDOWS_DEVICE_CACHE_MS = 60 * 1000;
let windowsDxdiagCachedAt = 0;
let windowsDxdiagCached: WindowsIntelDxdiagEntry[] = [];
let windowsDeviceCachedAt = 0;
let windowsDeviceCached: WindowsIntelDevices = { gpus: [], npuModels: [] };
let windowsGpuLuidByModel = new Map<string, string>();

function clampPercent(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function round1(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.round(value * 10) / 10;
}

function parseNumber(raw: unknown) {
  const value = Number(String(raw ?? '').trim());
  return Number.isFinite(value) ? value : null;
}

function extractLuidToken(raw: string) {
  const matched = String(raw || '').toLowerCase().match(/luid_0x[0-9a-f]+_0x[0-9a-f]+/);
  return matched ? matched[0] : '';
}

function extractPhysIndex(raw: string) {
  const matched = String(raw || '').toLowerCase().match(/_phys_(\d+)/);
  if (!matched) return Number.POSITIVE_INFINITY;
  const value = Number(matched[1]);
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

function toArray<T = unknown>(input: unknown): T[] {
  if (Array.isArray(input)) return input as T[];
  if (input == null) return [];
  return [input as T];
}

function uniqueStrings(values: string[]) {
  return Array.from(new Set(values.map((v) => String(v || '').trim()).filter(Boolean)));
}

function normalizeModelKey(value: string) {
  return String(value || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function inferWindowsGpuMemoryProfile(model: string, dedicatedGB?: number) {
  const normalized = normalizeModelKey(model);
  if (/arc\b/.test(normalized)) return 'discrete';
  if (dedicatedGB != null && dedicatedGB >= 1) return 'discrete';
  return 'shared';
}

function compareWindowsGpuTaskManagerOrder(
  a: { model: string; dedicatedGB?: number; taskManagerOrder?: number },
  b: { model: string; dedicatedGB?: number; taskManagerOrder?: number }
) {
  const orderA = a.taskManagerOrder ?? Number.POSITIVE_INFINITY;
  const orderB = b.taskManagerOrder ?? Number.POSITIVE_INFINITY;
  if (orderA !== orderB) return orderA - orderB;

  const profileA = inferWindowsGpuMemoryProfile(a.model, a.dedicatedGB);
  const profileB = inferWindowsGpuMemoryProfile(b.model, b.dedicatedGB);
  if (profileA !== profileB) return profileA === 'discrete' ? -1 : 1;

  const dedicatedA = a.dedicatedGB || 0;
  const dedicatedB = b.dedicatedGB || 0;
  if (dedicatedA !== dedicatedB) return dedicatedB - dedicatedA;

  return a.model.localeCompare(b.model);
}

function pickBestWindowsMetric(
  device: WindowsIntelDevices['gpus'][number] | undefined,
  metrics: WindowsGpuMetric[]
) {
  if (metrics.length === 0) return null;
  const model = device?.model || '';
  const modelKey = normalizeModelKey(model);
  const profile = inferWindowsGpuMemoryProfile(model, device?.dedicatedGB);
  const cachedLuid = windowsGpuLuidByModel.get(modelKey);
  if (cachedLuid) {
    const cachedMetric = metrics.find((metric) => metric.luid === cachedLuid);
    if (
      cachedMetric &&
      (profile === 'discrete'
        ? (cachedMetric.dedicatedUsedGB || 0) >= 0.05 || cachedMetric.engineTypes.includes('compute') || cachedMetric.engineTypes.includes('gsc')
        : (cachedMetric.dedicatedUsedGB || 0) <= 0.25)
    ) {
      return cachedMetric;
    }
  }

  let bestMetric: WindowsGpuMetric | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const metric of metrics) {
    const dedicated = metric.dedicatedUsedGB || 0;
    const shared = metric.sharedUsedGB || 0;
    const hasCompute = metric.engineTypes.includes('compute');
    const hasGsc = metric.engineTypes.includes('gsc');
    const score =
      profile === 'discrete'
        ? dedicated * 40 + metric.utilization * 0.05 + (hasCompute ? 4 : 0) + (hasGsc ? 2 : 0) - shared * 0.35
        : shared * 40 + metric.utilization * 0.05 - dedicated * 60 - (hasCompute ? 6 : 0) - (hasGsc ? 3 : 0);

    if (score > bestScore) {
      bestScore = score;
      bestMetric = metric;
    }
  }

  if (bestMetric && modelKey) {
    windowsGpuLuidByModel.set(modelKey, bestMetric.luid);
  }
  return bestMetric;
}

function parseDxdiagMemoryMB(line: string, labels: string[]) {
  const normalized = String(line || '').trim();
  if (!normalized) return null;
  if (!labels.some((label) => normalized.toLowerCase().includes(label.toLowerCase()))) return null;
  const matched = normalized.match(/:\s*([0-9]+)\s*MB/i);
  if (!matched) return null;
  const mb = Number(matched[1]);
  return Number.isFinite(mb) ? mb : null;
}

async function readWindowsIntelDxdiagEntries() {
  const now = Date.now();
  if (windowsDxdiagCached.length > 0 && now - windowsDxdiagCachedAt < WINDOWS_DXDIAG_CACHE_MS) {
    return windowsDxdiagCached;
  }

  const tmpPath = path.join(PathManager.getTmpPath(), `arcsub-dxdiag-${process.pid}-${Date.now()}.txt`);
  try {
    await execFileAsync('dxdiag', ['/t', tmpPath], { windowsHide: true });
    const text = await fs.readFile(tmpPath, 'utf8');
    const lines = text.split(/\r?\n/);

    type Entry = { model: string; dedicatedMB?: number; totalMB?: number };
    const entries: Entry[] = [];
    let current: Entry = { model: '' };

    const flush = () => {
      if (current.model || current.dedicatedMB || current.totalMB) {
        entries.push(current);
      }
      current = { model: '' };
    };

    for (const rawLine of lines) {
      const line = String(rawLine || '').trim();
      if (!line) continue;

      const cardMatch = line.match(/^Card name\s*:\s*(.+)$/i);
      if (cardMatch) {
        flush();
        current.model = String(cardMatch[1] || '').trim();
        continue;
      }

      const dedicatedMB = parseDxdiagMemoryMB(line, ['Dedicated Memory', 'Dedicated Video Memory']);
      if (dedicatedMB != null) {
        current.dedicatedMB = dedicatedMB;
        continue;
      }

      const totalMB = parseDxdiagMemoryMB(line, ['Display Memory', 'Approx. Total Memory', 'Total Memory']);
      if (totalMB != null) {
        current.totalMB = totalMB;
      }
    }
    flush();

    windowsDxdiagCached = entries
      .filter((entry) => /intel/i.test(entry.model || ''))
      .map((entry) => ({
        model: entry.model,
        dedicatedGB: entry.dedicatedMB ? entry.dedicatedMB / 1024 : undefined,
        totalGB: entry.totalMB ? entry.totalMB / 1024 : undefined,
      }));
    windowsDxdiagCachedAt = now;
    return windowsDxdiagCached;
  } catch {
    windowsDxdiagCachedAt = now;
    windowsDxdiagCached = [];
    return windowsDxdiagCached;
  } finally {
    try {
      await fs.unlink(tmpPath);
    } catch {
      // no-op
    }
  }
}

async function readWindowsIntelDevices() {
  const now = Date.now();
  if (now - windowsDeviceCachedAt < WINDOWS_DEVICE_CACHE_MS) {
    return windowsDeviceCached;
  }

  try {
    const script =
      "$gpu=Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name,PNPDeviceID; " +
      "$reg=@(); " +
      "$base='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'; " +
      "Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object { " +
      "  if ($_.PSChildName -notmatch '^\\d{4}$') { return }; " +
      "  try { " +
      "    $p=Get-ItemProperty $_.PSPath -ErrorAction Stop; " +
      "    $rawName=$p.'HardwareInformation.AdapterString'; " +
      "    $name=''; " +
      "    if ($rawName -is [byte[]]) { $name=[Text.Encoding]::Unicode.GetString($rawName).Trim([char]0) } else { $name=[string]$rawName }; " +
      "    $mem=$p.'HardwareInformation.qwMemorySize'; " +
      "    $order=0; " +
      "    [void][int]::TryParse($_.PSChildName, [ref]$order); " +
      "    if ($name) { $reg += [pscustomobject]@{ Name=$name; MemoryBytes=$mem; Order=$order } } " +
      "  } catch {} " +
      "}; " +
      "$npu=Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | " +
      "Where-Object { ($_.Name -match 'NPU|Neural|AI Boost|VPU') -and (($_.Manufacturer -match 'Intel') -or ($_.Name -match 'Intel')) } | Select-Object Name; " +
      "[pscustomobject]@{ gpu=@($gpu); reg=@($reg); npu=@($npu) } | ConvertTo-Json -Depth 8 -Compress";
    const { stdout } = await execFileAsync('powershell', ['-NoProfile', '-Command', script], { windowsHide: true });
    const payloadText = String(stdout || '').trim();
    const payload = payloadText ? JSON.parse(payloadText) : {};
    const regByModel = new Map<string, { model: string; dedicatedGB?: number; taskManagerOrder?: number }>();
    for (const entry of toArray<{ Name?: string; MemoryBytes?: number; Order?: number }>(payload?.reg)) {
      const model = String(entry?.Name || '').trim();
      if (!/intel/i.test(model)) continue;
      const key = normalizeModelKey(model);
      const prev = regByModel.get(key);
      const memoryBytes = Number(entry?.MemoryBytes || 0);
      const memoryGB =
        Number.isFinite(memoryBytes) && memoryBytes > 0
          ? memoryBytes / (1024 ** 3)
          : undefined;
      const taskManagerOrder = Number(entry?.Order);
      regByModel.set(key, {
        model,
        dedicatedGB:
          memoryGB != null
            ? prev?.dedicatedGB != null
              ? Math.max(prev.dedicatedGB, memoryGB)
              : memoryGB
            : prev?.dedicatedGB,
        taskManagerOrder:
          Number.isFinite(taskManagerOrder)
            ? prev?.taskManagerOrder != null
              ? Math.min(prev.taskManagerOrder, taskManagerOrder)
              : taskManagerOrder
            : prev?.taskManagerOrder,
      });
    }

    const gpuModels = uniqueStrings(
      toArray<{ Name?: string }>(payload?.gpu)
        .map((item) => String(item?.Name || '').trim())
        .filter((name) => /intel/i.test(name))
    );
    const gpus = gpuModels.map((model) => ({
      model,
      dedicatedGB: regByModel.get(normalizeModelKey(model))?.dedicatedGB,
      taskManagerOrder: regByModel.get(normalizeModelKey(model))?.taskManagerOrder,
    }));

    // Include Intel adapters that only appear in registry enumeration.
    for (const [key, info] of regByModel.entries()) {
      if (gpus.some((item) => normalizeModelKey(item.model) === key)) continue;
      gpus.push({
        model: info.model,
        dedicatedGB: info.dedicatedGB,
        taskManagerOrder: info.taskManagerOrder,
      });
    }
    gpus.sort(compareWindowsGpuTaskManagerOrder);

    const npuModels = uniqueStrings(
      toArray<{ Name?: string }>(payload?.npu)
        .map((item) => String(item?.Name || '').trim())
        .filter((name) => /intel/i.test(name))
    );
    windowsDeviceCached = { gpus, npuModels };
  } catch {
    windowsDeviceCached = { gpus: [], npuModels: [] };
  }

  windowsDeviceCachedAt = now;
  return windowsDeviceCached;
}

function toCpuTicksSnapshot() {
  const cpus = os.cpus();
  if (!cpus.length) return { idle: 0, total: 0 };
  let idle = 0;
  let total = 0;
  for (const cpu of cpus) {
    idle += cpu.times.idle;
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq;
  }
  return { idle, total };
}

async function detectWindowsAccelerators() {
  type EngineRow = { InstanceName?: string; CookedValue?: number };
  type MemoryRow = { Path?: string; CookedValue?: number };
  type LuidSlot = {
    physIndex: number;
    values: Map<string, number>;
    engineTypes: Set<string>;
    dedicatedBytes: number;
    sharedBytes: number;
    committedBytes: number;
    hasMemorySample: boolean;
  };

  try {
    const [devices, dxEntries] = await Promise.all([
      readWindowsIntelDevices(),
      readWindowsIntelDxdiagEntries(),
    ]);
    const dxByModel = new Map(dxEntries.map((entry) => [normalizeModelKey(entry.model), entry]));
    const remainingDxEntries = [...dxEntries].sort(compareWindowsGpuTaskManagerOrder);

    const script =
      "$eng=Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue; " +
      "$mem=Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Total Committed' -ErrorAction SilentlyContinue; " +
      "$npu=$null; try { $npu=Get-Counter '\\NPU Engine(*)\\Utilization Percentage' -ErrorAction Stop } catch { } ; " +
      "[pscustomobject]@{ engines=@($eng.CounterSamples | Select-Object InstanceName,CookedValue); memory=@($mem.CounterSamples | Select-Object Path,CookedValue); npuEngines=@($npu.CounterSamples | Select-Object InstanceName,CookedValue) } | ConvertTo-Json -Depth 6 -Compress";
    const { stdout } = await execFileAsync('powershell', ['-NoProfile', '-Command', script], { windowsHide: true });
    const payloadText = String(stdout || '').trim();
    const payload = payloadText ? JSON.parse(payloadText) : {};

    const engineRows = toArray<EngineRow>(payload?.engines);
    const memoryRows = toArray<MemoryRow>(payload?.memory);
    const npuRows = toArray<EngineRow>(payload?.npuEngines);

    const byLuid = new Map<string, LuidSlot>();
    for (const row of engineRows) {
      const name = String(row?.InstanceName || '').toLowerCase();
      const luid = extractLuidToken(name);
      if (!luid) continue;
      const value = Number(row?.CookedValue || 0);
      if (!Number.isFinite(value)) continue;

      const slot = byLuid.get(luid) || {
        physIndex: Number.POSITIVE_INFINITY,
        values: new Map<string, number>(),
        engineTypes: new Set<string>(),
        dedicatedBytes: 0,
        sharedBytes: 0,
        committedBytes: 0,
        hasMemorySample: false,
      };
      slot.physIndex = Math.min(slot.physIndex, extractPhysIndex(name));

      const engineTypeMatched = name.match(/engtype_([a-z0-9]+)/);
      const engineType = engineTypeMatched?.[1] || '';
      const engineKey = name.replace(/^pid_\d+_/, '');
      slot.values.set(engineKey, Math.max(slot.values.get(engineKey) || 0, value));
      if (engineType) slot.engineTypes.add(engineType);
      byLuid.set(luid, slot);
    }

    for (const row of memoryRows) {
      const p = String(row?.Path || '').toLowerCase();
      const luid = extractLuidToken(p);
      if (!luid) continue;
      const value = Number(row?.CookedValue || 0);
      if (!Number.isFinite(value)) continue;

      const slot = byLuid.get(luid) || {
        physIndex: Number.POSITIVE_INFINITY,
        values: new Map<string, number>(),
        engineTypes: new Set<string>(),
        dedicatedBytes: 0,
        sharedBytes: 0,
        committedBytes: 0,
        hasMemorySample: false,
      };
      slot.hasMemorySample = true;
      if (p.includes('dedicated usage')) slot.dedicatedBytes = Math.max(slot.dedicatedBytes, value);
      if (p.includes('shared usage')) slot.sharedBytes = Math.max(slot.sharedBytes, value);
      if (p.includes('total committed')) slot.committedBytes = Math.max(slot.committedBytes, value);
      byLuid.set(luid, slot);
    }

    const preferredTypes = ['3d', 'compute', 'videodecode', 'videoprocessing', 'copy', 'gsc'];
    const metrics = Array.from(byLuid.entries()).map(([luid, slot]) => {
      const selectedValues = Array.from(slot.values.entries())
        .filter(([key]) => preferredTypes.some((type) => key.includes(`engtype_${type}`)))
        .map(([, value]) => value);
      const values3d = Array.from(slot.values.entries())
        .filter(([key]) => key.includes('engtype_3d'))
        .map(([, value]) => value);
      const valuesCompute = Array.from(slot.values.entries())
        .filter(([key]) => key.includes('engtype_compute'))
        .map(([, value]) => value);
      const values = selectedValues.length > 0 ? selectedValues : Array.from(slot.values.values());
      const utilization = values.length > 0 ? Math.max(...values) : 0;
      const utilization3d = values3d.length > 0 ? Math.max(...values3d) : 0;
      const utilizationCompute = valuesCompute.length > 0 ? Math.max(...valuesCompute) : 0;
      const dedicatedUsedGB = slot.hasMemorySample ? slot.dedicatedBytes / (1024 ** 3) : undefined;
      const sharedUsedGB = slot.hasMemorySample ? slot.sharedBytes / (1024 ** 3) : undefined;
      return {
        luid,
        physIndex: slot.physIndex,
        utilization,
        utilization3d,
        utilizationCompute,
        engineTypes: Array.from(slot.engineTypes),
        dedicatedUsedGB,
        sharedUsedGB,
      };
    });

    metrics.sort((a, b) => {
      if (a.physIndex !== b.physIndex) return a.physIndex - b.physIndex;
      return b.utilization - a.utilization;
    });

    const accelerators: AcceleratorSnapshot[] = [];
    const count = devices.gpus.length > 0 ? devices.gpus.length : metrics.length;
    const remainingMetrics = [...metrics];
    const sharedVramTotalGB = os.totalmem() / (1024 ** 3) / 2;
    for (let i = 0; i < count; i += 1) {
      const device = devices.gpus[i];
      const metric =
        device != null
          ? pickBestWindowsMetric(device, remainingMetrics)
          : remainingMetrics.length > 0
            ? remainingMetrics.shift() || null
            : null;
      if (metric) {
        const metricIndex = remainingMetrics.findIndex((item) => item.luid === metric.luid);
        if (metricIndex >= 0) remainingMetrics.splice(metricIndex, 1);
      }
      const model = device?.model || remainingDxEntries.shift()?.model || `Intel GPU ${i + 1}`;
      const modelKey = normalizeModelKey(model);
      const registryVramTotalGB = device?.dedicatedGB;
      const matchedDxEntry = modelKey ? dxByModel.get(modelKey) : undefined;
      const dxVramTotalGB = matchedDxEntry?.dedicatedGB || matchedDxEntry?.totalGB;
      const dedicatedVramTotalGB =
        registryVramTotalGB && registryVramTotalGB > 0
          ? registryVramTotalGB
          : dxVramTotalGB && dxVramTotalGB > 0
            ? dxVramTotalGB
            : undefined;
      const profile = inferWindowsGpuMemoryProfile(model, dedicatedVramTotalGB);
      const has3dOrComputeMetric =
        metric != null &&
        ((metric.utilization3d > 0 || metric.engineTypes.includes('3d')) ||
          (metric.utilizationCompute > 0 || metric.engineTypes.includes('compute')));
      const utilization =
        profile === 'shared' && has3dOrComputeMetric
          ? Math.max(metric?.utilization3d || 0, metric?.utilizationCompute || 0)
          : metric?.utilization || 0;
      const vramUsedGB = profile === 'discrete' ? metric?.dedicatedUsedGB : metric?.sharedUsedGB;
      const vramTotalGB =
        profile === 'discrete'
          ? dedicatedVramTotalGB
          : dedicatedVramTotalGB && dedicatedVramTotalGB >= 0.5
            ? Math.max(sharedVramTotalGB, dedicatedVramTotalGB)
            : sharedVramTotalGB;
      if (metric && modelKey) {
        windowsGpuLuidByModel.set(modelKey, metric.luid);
      }
      accelerators.push({
        id: `windows-gpu-${i}`,
        kind: 'gpu',
        vendor: 'intel',
        model,
        utilization,
        vramUsedGB,
        vramTotalGB,
        luid: metric?.luid,
        memorySource: profile === 'discrete' ? 'dedicated' : 'shared',
        engineTypes: metric?.engineTypes || [],
        physIndex: metric?.physIndex,
        taskManagerIndex: i,
      });
    }

    const npuUtilValues = npuRows
      .map((row) => Number(row?.CookedValue || 0))
      .filter((value) => Number.isFinite(value));
    const npuUtilization = npuUtilValues.length > 0 ? Math.max(...npuUtilValues) : 0;
    devices.npuModels.forEach((model, index) => {
      accelerators.push({
        id: `windows-npu-${index}`,
        kind: 'npu',
        vendor: 'intel',
        model,
        utilization: npuUtilization,
      });
    });

    return accelerators;
  } catch {
    return [] as AcceleratorSnapshot[];
  }
}

async function detectLinuxAccelerators() {
  const accelerators: AcceleratorSnapshot[] = [];

  try {
    const drmRoot = '/sys/class/drm';
    const entries = await fs.readdir(drmRoot);
    const cards = entries.filter((entry) => /^card\d+$/.test(entry));

    for (const card of cards) {
      const deviceDir = path.join(drmRoot, card, 'device');
      let isIntel = false;
      try {
        const vendorRaw = await fs.readFile(path.join(deviceDir, 'vendor'), 'utf8');
        isIntel = String(vendorRaw || '').trim().toLowerCase() === '0x8086';
      } catch {
        isIntel = false;
      }
      if (!isIntel) continue;

      let driverName = '';
      try {
        driverName = path.basename(await fs.readlink(path.join(deviceDir, 'driver')));
      } catch {
        driverName = '';
      }
      const model = driverName ? `Intel Graphics (${driverName}, ${card})` : `Intel Graphics (${card})`;

      let utilization = 0;
      const utilCandidates = [
        path.join(deviceDir, 'gt_busy_percent'),
        path.join(deviceDir, 'gpu_busy_percent'),
        path.join(drmRoot, card, 'gt_busy_percent'),
      ];
      for (const utilPath of utilCandidates) {
        try {
          const utilRaw = await fs.readFile(utilPath, 'utf8');
          const util = parseNumber(utilRaw);
          if (util != null) {
            utilization = util;
            break;
          }
        } catch {
          // continue
        }
      }

      let vramUsedGB: number | undefined;
      let vramTotalGB: number | undefined;
      const memoryCandidates: Array<{ used: string; total: string }> = [
        { used: path.join(deviceDir, 'mem_info_vram_used'), total: path.join(deviceDir, 'mem_info_vram_total') },
        { used: path.join(deviceDir, 'lmem_used_bytes'), total: path.join(deviceDir, 'lmem_total_bytes') },
      ];
      for (const candidate of memoryCandidates) {
        try {
          const usedRaw = await fs.readFile(candidate.used, 'utf8');
          const totalRaw = await fs.readFile(candidate.total, 'utf8');
          const used = parseNumber(usedRaw);
          const total = parseNumber(totalRaw);
          if (used != null && total != null && total > 0) {
            vramUsedGB = used / (1024 ** 3);
            vramTotalGB = total / (1024 ** 3);
            break;
          }
        } catch {
          // continue
        }
      }

      accelerators.push({
        id: `linux-gpu-${card}`,
        kind: 'gpu',
        vendor: 'intel',
        model,
        utilization,
        vramUsedGB,
        vramTotalGB,
      });
    }
  } catch {
    // ignore gpu probe errors
  }

  try {
    const accelRoot = '/sys/class/accel';
    const entries = await fs.readdir(accelRoot);
    const npus = entries.filter((entry) => /^accel\d+$/.test(entry));
    for (const accel of npus) {
      const deviceDir = path.join(accelRoot, accel, 'device');
      let isIntel = false;
      try {
        const vendorRaw = await fs.readFile(path.join(deviceDir, 'vendor'), 'utf8');
        isIntel = String(vendorRaw || '').trim().toLowerCase() === '0x8086';
      } catch {
        isIntel = false;
      }
      if (!isIntel) continue;

      let driverName = '';
      try {
        driverName = path.basename(await fs.readlink(path.join(deviceDir, 'driver')));
      } catch {
        driverName = '';
      }
      const model = driverName ? `Intel NPU (${driverName}, ${accel})` : `Intel NPU (${accel})`;

      let utilization = 0;
      const utilCandidates = [
        path.join(accelRoot, accel, 'busy_percent'),
        path.join(accelRoot, accel, 'utilization'),
        path.join(deviceDir, 'busy_percent'),
      ];
      for (const utilPath of utilCandidates) {
        try {
          const utilRaw = await fs.readFile(utilPath, 'utf8');
          const util = parseNumber(utilRaw);
          if (util != null) {
            utilization = util;
            break;
          }
        } catch {
          // continue
        }
      }

      accelerators.push({
        id: `linux-npu-${accel}`,
        kind: 'npu',
        vendor: 'intel',
        model,
        utilization,
      });
    }
  } catch {
    // ignore npu probe errors
  }

  return accelerators;
}

async function detectAccelerators() {
  if (process.platform === 'win32') {
    return detectWindowsAccelerators();
  }
  if (process.platform === 'linux') {
    return detectLinuxAccelerators();
  }
  return [] as AcceleratorSnapshot[];
}

function normalizeAccelerators(accelerators: AcceleratorSnapshot[]) {
  const normalized = accelerators.map((item) => {
    const vramUsedGB =
      item.vramUsedGB != null ? Math.max(0, item.vramUsedGB) : undefined;
    const rawVramTotalGB =
      item.vramTotalGB != null && item.vramTotalGB > 0 ? Math.max(0, item.vramTotalGB) : undefined;
    const safeVramTotalGB =
      rawVramTotalGB != null && vramUsedGB != null && vramUsedGB > rawVramTotalGB ? vramUsedGB : rawVramTotalGB;
    return {
      ...item,
      utilization: clampPercent(item.utilization),
      vramUsedGB: vramUsedGB != null ? round1(vramUsedGB) : undefined,
      vramTotalGB: safeVramTotalGB != null ? round1(safeVramTotalGB) : undefined,
    };
  });

  let fallbackGpuIndex = 0;
  return normalized.map((item) => {
    if (item.kind !== 'gpu') return item;
    const taskManagerIndex = item.taskManagerIndex ?? fallbackGpuIndex;
    fallbackGpuIndex += 1;
    return {
      ...item,
      taskManagerIndex,
    };
  });
}

export class SystemMonitor {
  private static readonly CACHE_MS = 3000;
  private static readonly FAST_CACHE_MS = 1000;
  private static lastCpuSnapshot: CpuTicksSnapshot | null = null;
  private static cachedAt = 0;
  private static cached: SystemResourceSnapshot | null = null;
  private static inFlightSnapshot: Promise<SystemResourceSnapshot> | null = null;

  private static async refreshSnapshot(): Promise<SystemResourceSnapshot> {
    const cpuPercent = this.readCpuPercent();
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = Math.max(0, totalMem - freeMem);
    const ramPercent = totalMem > 0 ? (usedMem / totalMem) * 100 : 0;

    const accelerators = normalizeAccelerators(await detectAccelerators());
    const gpuAccelerators = accelerators.filter((item) => item.kind === 'gpu');
    const primaryGpu = gpuAccelerators[0];
    const primaryVramPercent =
      primaryGpu?.vramTotalGB && primaryGpu.vramTotalGB > 0 && primaryGpu.vramUsedGB != null
        ? clampPercent((primaryGpu.vramUsedGB / primaryGpu.vramTotalGB) * 100)
        : 0;

    const cpuModel = String(os.cpus()?.[0]?.model || 'Unknown CPU').trim();

    this.cached = {
      cpu: clampPercent(cpuPercent),
      ram: clampPercent(ramPercent),
      gpu: primaryGpu?.utilization || 0,
      vram: primaryVramPercent,
      ramUsedGB: round1(usedMem / (1024 ** 3)),
      ramTotalGB: round1(totalMem / (1024 ** 3)),
      vramUsedGB: primaryGpu?.vramUsedGB,
      vramTotalGB: primaryGpu?.vramTotalGB,
      platform: process.platform,
      cpuModel,
      gpuModel: primaryGpu?.model || 'Intel Graphics',
      accelerators,
    };
    this.cachedAt = Date.now();
    return this.cached;
  }

  static async getSnapshot(forceFresh = false, preferHighFrequency = false): Promise<SystemResourceSnapshot> {
    const cacheMs = preferHighFrequency ? this.FAST_CACHE_MS : this.CACHE_MS;
    const now = Date.now();
    if (!forceFresh && this.cached && now - this.cachedAt < cacheMs) {
      return this.cached;
    }

    if (!forceFresh && this.cached) {
      if (!this.inFlightSnapshot) {
        const backgroundTask = this.refreshSnapshot();
        this.inFlightSnapshot = backgroundTask;
        void backgroundTask.finally(() => {
          if (this.inFlightSnapshot === backgroundTask) {
            this.inFlightSnapshot = null;
          }
        });
      }
      return this.cached;
    }

    if (!forceFresh && this.inFlightSnapshot) {
      return this.inFlightSnapshot;
    }

    const snapshotTask = this.refreshSnapshot();
    this.inFlightSnapshot = snapshotTask;
    try {
      return await snapshotTask;
    } finally {
      if (this.inFlightSnapshot === snapshotTask) {
        this.inFlightSnapshot = null;
      }
    }
  }

  private static readCpuPercent() {
    const current = toCpuTicksSnapshot();
    if (!this.lastCpuSnapshot) {
      this.lastCpuSnapshot = current;
      return 0;
    }

    const idleDiff = current.idle - this.lastCpuSnapshot.idle;
    const totalDiff = current.total - this.lastCpuSnapshot.total;
    this.lastCpuSnapshot = current;

    if (totalDiff <= 0) return 0;
    return 100 - (idleDiff / totalDiff) * 100;
  }
}
