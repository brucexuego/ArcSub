process.env.ARCSUB_RELEASE_TARGET = process.env.ARCSUB_RELEASE_TARGET || 'linux-x64';
await import('./package-release.mjs');
