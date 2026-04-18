import fs from 'fs-extra';
import path from 'node:path';
import { PathManager } from '../server/path_manager.js';

type CleanupTarget = 'tmp' | 'logs' | 'cache';

interface CliOptions {
  apply: boolean;
  targets: Set<CleanupTarget>;
}

interface CleanupAction {
  target: CleanupTarget;
  path: string;
  existed: boolean;
  removed: boolean;
  skippedReason: string | null;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    apply: false,
    targets: new Set<CleanupTarget>(['tmp', 'logs', 'cache']),
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = String(argv[index] || '').trim();
    const next = String(argv[index + 1] || '').trim();
    switch (arg) {
      case '--apply':
        options.apply = true;
        break;
      case '--target': {
        if (next === 'tmp' || next === 'logs' || next === 'cache' || next === 'all') {
          options.targets = next === 'all' ? new Set<CleanupTarget>(['tmp', 'logs', 'cache']) : new Set<CleanupTarget>([next]);
          index += 1;
        }
        break;
      }
      default:
        break;
    }
  }

  return options;
}

function isPathInsideOrSame(parent: string, child: string) {
  const normalizedParent = path.resolve(parent);
  const normalizedChild = path.resolve(child);
  const rel = path.relative(normalizedParent, normalizedChild);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

async function removePathSafely(input: {
  target: CleanupTarget;
  targetPath: string;
  runtimeRoot: string;
  apply: boolean;
}): Promise<CleanupAction> {
  const resolved = path.resolve(input.targetPath);
  if (!isPathInsideOrSame(input.runtimeRoot, resolved)) {
    return {
      target: input.target,
      path: resolved,
      existed: await fs.pathExists(resolved),
      removed: false,
      skippedReason: 'Path escapes runtime root',
    };
  }

  const existed = await fs.pathExists(resolved);
  if (!existed) {
    return {
      target: input.target,
      path: resolved,
      existed,
      removed: false,
      skippedReason: 'Path not found',
    };
  }

  if (!input.apply) {
    return {
      target: input.target,
      path: resolved,
      existed,
      removed: false,
      skippedReason: 'Dry run',
    };
  }

  await fs.remove(resolved);
  await fs.ensureDir(resolved);
  return {
    target: input.target,
    path: resolved,
    existed,
    removed: true,
    skippedReason: null,
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const runtimeRoot = path.resolve(PathManager.getRuntimePath());

  const actions: CleanupAction[] = [];
  if (options.targets.has('tmp')) {
    actions.push(
      await removePathSafely({
        target: 'tmp',
        targetPath: PathManager.getTmpPath(),
        runtimeRoot,
        apply: options.apply,
      })
    );
  }

  if (options.targets.has('logs')) {
    actions.push(
      await removePathSafely({
        target: 'logs',
        targetPath: PathManager.getLogsPath(),
        runtimeRoot,
        apply: options.apply,
      })
    );
  }

  if (options.targets.has('cache')) {
    actions.push(
      await removePathSafely({
        target: 'cache',
        targetPath: path.join(PathManager.getModelsPath(), 'openvino-cache'),
        runtimeRoot,
        apply: options.apply,
      })
    );
  }

  console.log(
    JSON.stringify(
      {
        mode: options.apply ? 'apply' : 'dry-run',
        runtimeRoot,
        actions,
      },
      null,
      2
    )
  );
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2
    )
  );
  process.exitCode = 1;
});
