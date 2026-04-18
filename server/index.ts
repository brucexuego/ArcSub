import 'dotenv/config';
import { registerProcessErrorLogging, resolveServerListenConfig, startServer, writeAsrLog } from '../server.js';
registerProcessErrorLogging();

startServer(resolveServerListenConfig()).catch((error) => {
  writeAsrLog('[Server startup failed]', error?.stack || error?.message || String(error || 'Unknown startup error'), 'error');
  process.exitCode = 1;
});
