import assert from "node:assert/strict";
import test from "node:test";

import {
  SETTINGS_SCHEMA_VERSION,
  createSettingsStore,
} from "../fsrcnnx-settings-store.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function clone(value) {
  return value === undefined ? undefined : structuredClone(value);
}

function fieldKey(scope, field) {
  return `fsrcnnx_setting:${encodeURIComponent(scope)}:${encodeURIComponent(field)}`;
}

function schemaKey(scope) {
  return `fsrcnnx_setting:${encodeURIComponent(scope)}:$schema`;
}

function record(value, { source = "seed", time = 1, counter = 0 } = {}) {
  const result = {
    schemaVersion: SETTINGS_SCHEMA_VERSION,
    source,
    stamp: { time, counter },
  };
  if (value === undefined) result.deleted = true;
  else result.value = value;
  return result;
}

function clock(start) {
  let current = start;
  return () => current++;
}

async function settle(turns = 4) {
  for (let i = 0; i < turns; i++) await new Promise((resolve) => setImmediate(resolve));
}

class ChangeEvent {
  constructor() { this.listeners = new Set(); }
  addListener(listener) { this.listeners.add(listener); }
  removeListener(listener) { this.listeners.delete(listener); }
  emit(changes, areaName = "local") {
    for (const listener of [...this.listeners]) listener(clone(changes), areaName);
  }
}

class MemoryStorage {
  constructor(seed = {}) {
    this.data = clone(seed);
    this.onChanged = new ChangeEvent();
    this.getCalls = [];
    this.setCalls = [];
    this.getPlans = [];
    this.setPlans = [];
  }

  planGet(plan) { this.getPlans.push(plan); }
  planSet(plan) { this.setPlans.push(plan); }

  async get(keys) {
    this.getCalls.push([...keys]);
    const result = {};
    for (const key of keys) if (Object.prototype.hasOwnProperty.call(this.data, key)) result[key] = clone(this.data[key]);
    const plan = this.getPlans.shift();
    if (plan?.gate) await plan.gate.promise;
    if (plan?.error) throw plan.error;
    return result;
  }

  async set(items) {
    const saved = clone(items);
    this.setCalls.push(saved);
    const plan = this.setPlans.shift();
    if (plan?.gate) await plan.gate.promise;
    if (plan?.error) throw plan.error;
    this.commit(saved, { emit: plan?.emit !== false });
  }

  commit(items, { emit = true, areaName = "local" } = {}) {
    const changes = {};
    for (const [key, value] of Object.entries(items)) {
      const oldValue = clone(this.data[key]);
      this.data[key] = clone(value);
      changes[key] = { oldValue, newValue: clone(value) };
    }
    if (emit && Object.keys(changes).length) this.onChanged.emit(changes, areaName);
  }
}

function seededStorage(scope, fields = {}) {
  const seed = { [schemaKey(scope)]: { schemaVersion: SETTINGS_SCHEMA_VERSION } };
  for (const [field, value] of Object.entries(fields)) seed[fieldKey(scope, field)] = value;
  return new MemoryStorage(seed);
}

test("same-scope contexts update unrelated fields without lost full-snapshot writes", async () => {
  const scope = "https://video.example";
  const storage = seededStorage(scope);
  const fields = ["mode", "policy"];
  const externalA = [];
  const externalB = [];
  const a = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields, sourceId: "tab-a", now: clock(100),
  });
  const b = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields, sourceId: "tab-b", now: clock(200),
  });
  await Promise.all([a.ready, b.ready]);
  a.subscribe((patch) => externalA.push(patch));
  b.subscribe((patch) => externalB.push(patch));

  await Promise.all([
    a.write({ mode: "upscale" }),
    b.write({ policy: "force4" }),
  ]);

  assert.equal(storage.setCalls.length, 2);
  assert.deepEqual(storage.setCalls.map((call) => Object.keys(call)), [
    [fieldKey(scope, "mode")],
    [fieldKey(scope, "policy")],
  ]);
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.equal(storage.data[fieldKey(scope, "policy")].value, "force4");
  assert.deepEqual(a.snapshot(), { mode: "upscale", policy: "force4" });
  assert.deepEqual(b.snapshot(), { mode: "upscale", policy: "force4" });
  assert.deepEqual(externalA, [{ policy: "force4" }], "own mode event must be suppressed");
  assert.deepEqual(externalB, [{ mode: "upscale" }], "own policy event must be suppressed");
  assert.deepEqual(a.health(), {
    state: "ready", operation: null, errorOperation: null, pending: 0, error: null,
    schemaVersion: SETTINGS_SCHEMA_VERSION, scope, closed: false,
  });
  a.close();
  b.close();
});

test("same-field writes are serialized and coalesce queued intent to the newest value", async () => {
  const scope = "https://ordered.example";
  const storage = seededStorage(scope);
  const firstWrite = deferred();
  storage.planSet({ gate: firstWrite });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "ordered", now: clock(10),
  });
  await store.ready;

  const first = store.write({ mode: "passthrough" });
  await settle();
  const second = store.write({ mode: "upscale" });
  const third = store.write({ mode: "off" });
  await settle();
  assert.equal(store.health().state, "writing");
  assert.equal(store.health().operation, "writing");
  assert.equal(store.health().pending, 1, "one field remains pending despite multiple revisions");

  firstWrite.resolve();
  await Promise.all([first, second, third]);
  assert.equal(storage.setCalls.length, 2, "the middle queued value is coalesced");
  assert.deepEqual(storage.setCalls.map((call) => call[fieldKey(scope, "mode")].value), [
    "passthrough", "off",
  ]);
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "off");
  assert.deepEqual(store.snapshot(), { mode: "off" });
  store.close();
});

test("flush waits for a write that has already moved into the in-flight batch", async () => {
  const scope = "https://flush-in-flight.example";
  const storage = seededStorage(scope);
  const committed = deferred();
  storage.planSet({ gate: committed });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"],
    sourceId: "flush-in-flight", now: clock(10),
  });
  await store.ready;

  const write = store.write({ mode: "upscale" });
  await settle();
  assert.equal(storage.setCalls.length, 1);
  assert.equal(storage.data[fieldKey(scope, "mode")], undefined,
    "the gated storage write has not committed");

  let flushSettled = false;
  const flush = store.flush().then((snapshot) => {
    flushSettled = true;
    return snapshot;
  });
  await settle();
  assert.equal(flushSettled, false, "flush must await the active storage.set call");
  assert.equal(store.health().pending, 1);

  committed.resolve();
  assert.deepEqual(await flush, { mode: "upscale" });
  await write;
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.equal(store.health().state, "ready");
  store.close();
});

test("a delayed older same-field write converges back to later intent from another context", async () => {
  const scope = "https://race.example";
  const storage = seededStorage(scope);
  const delayedOlderWrite = deferred();
  storage.planSet({ gate: delayedOlderWrite });
  const a = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "z-older", now: () => 100,
  });
  const b = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "a-later", now: () => 100,
  });
  await Promise.all([a.ready, b.ready]);

  const older = a.write({ mode: "passthrough" });
  await settle();
  await b.write({ mode: "upscale" });
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");

  delayedOlderWrite.resolve();
  await older;
  await settle(8);
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.deepEqual(a.snapshot(), { mode: "upscale" });
  assert.deepEqual(b.snapshot(), { mode: "upscale" });
  assert.equal(a.health().pending, 0);
  assert.equal(b.health().pending, 0);
  const settledWrites = storage.setCalls.length;
  await settle(12);
  assert.equal(storage.setCalls.length, settledWrites, "equal-stamp repair converges without ping-pong");
  a.close();
  b.close();
});

test("same-millisecond sequential intent follows observation order, not source lexicography", async () => {
  const scope = "https://same-millisecond.example";
  const storage = seededStorage(scope);
  const first = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "z-first", now: () => 500,
  });
  const last = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "a-last", now: () => 500,
  });
  await Promise.all([first.ready, last.ready]);

  await first.write({ mode: "passthrough" });
  await last.write({ mode: "upscale" });
  assert.equal(storage.setCalls[0][fieldKey(scope, "mode")].stamp.counter, 0);
  assert.equal(storage.setCalls[1][fieldKey(scope, "mode")].stamp.counter, 1,
    "observing the first commit advances the local hybrid counter in the same millisecond");
  assert.equal(storage.data[fieldKey(scope, "mode")].source, "a-last");
  assert.deepEqual(first.snapshot(), { mode: "upscale" });
  assert.deepEqual(last.snapshot(), { mode: "upscale" });
  first.close();
  last.close();
});

test("legacy layouts remain zero-write read-through until the first explicit v2 field write", async () => {
  const scope = "https://legacy.example:8443";
  const host = "legacy.example";
  const storage = new MemoryStorage({
    fsrcnnx_sites: {
      [host]: { mode: "passthrough", policy: "auto", ignored: "old" },
    },
    [`fsrcnnx_site:${encodeURIComponent(host)}`]: {
      mode: "upscale", sharpen: true, ignored: "current",
    },
    [fieldKey(scope, "mode")]: record("off", { source: "new-schema", time: 50 }),
  });
  const store = createSettingsStore({
    storage,
    onChanged: storage.onChanged,
    scope,
    legacyHosts: [host],
    fields: ["mode", "policy", "sharpen"],
    sourceId: "migrator",
    now: clock(100),
  });

  assert.deepEqual(await store.ready, { mode: "off", policy: "auto", sharpen: true });
  assert.equal(storage.setCalls.length, 0, "ready never materializes legacy data or a schema marker");

  await store.write({ sharpen: false });
  assert.equal(storage.setCalls.length, 1);
  const firstWrite = storage.setCalls[0];
  assert.deepEqual(Object.keys(firstWrite).sort(), [
    fieldKey(scope, "sharpen"),
    schemaKey(scope),
  ].sort(), "only the explicitly changed field accompanies the first v2 schema marker");
  assert.deepEqual(firstWrite[fieldKey(scope, "sharpen")], {
    schemaVersion: SETTINGS_SCHEMA_VERSION,
    source: "migrator",
    stamp: { time: 100, counter: 0 },
    value: false,
  });
  assert.equal(firstWrite[schemaKey(scope)].schemaVersion, SETTINGS_SCHEMA_VERSION);
  assert.deepEqual(store.snapshot(), { mode: "off", policy: "auto", sharpen: false });
  assert.equal(storage.data.fsrcnnx_sites[host].mode, "passthrough", "migration does not rewrite the shared legacy map");

  const second = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, legacyHosts: [host],
    fields: ["mode", "policy", "sharpen"], sourceId: "second", now: clock(500),
  });
  assert.deepEqual(await second.ready, { mode: "off", policy: "auto", sharpen: false },
    "missing v2 fields continue to read through legacy after the schema marker exists");
  assert.equal(storage.setCalls.length, 1, "read-through initialization always remains write-free");
  store.close();
  second.close();
});

test("retired deband fields are inert and cannot poison the active settings schema", async () => {
  const scope = "https://retired.example";
  const host = "retired.example";
  const retiredEnabledKey = fieldKey(scope, "deband");
  const retiredStrengthKey = fieldKey(scope, "debandStrength");
  const storage = new MemoryStorage({
    [schemaKey(scope)]: { schemaVersion: SETTINGS_SCHEMA_VERSION },
    [fieldKey(scope, "mode")]: record("upscale"),
    [retiredEnabledKey]: { malformed: true },
    [retiredStrengthKey]: record(Infinity),
    fsrcnnx_sites: { [host]: { deband: true, debandStrength: 2.5 } },
  });
  const patches = [];
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, legacyHosts: [host], sourceId: "retired-fields",
  });
  store.subscribe((patch) => patches.push(patch));

  assert.deepEqual(await store.ready, { mode: "upscale" });
  assert.equal(storage.getCalls[0].includes(retiredEnabledKey), false);
  assert.equal(storage.getCalls[0].includes(retiredStrengthKey), false);
  assert.equal(store.health().state, "ready", "malformed retired records are not active corruption");

  storage.commit({
    [retiredEnabledKey]: record(false, { source: "old-tab", time: 50 }),
    [retiredStrengthKey]: { malformed: "again" },
  });
  assert.deepEqual(store.snapshot(), { mode: "upscale" });
  assert.deepEqual(patches, [], "late writes from an old extension version are ignored");
  assert.equal(storage.setCalls.length, 0, "retirement is zero-write and leaves unrelated storage untouched");
  store.close();
});

test("a corrupt explicit v2 field blocks legacy fallback and reports bounded health", async () => {
  assert.equal(SETTINGS_SCHEMA_VERSION, 2);
  const scope = "https://corrupt.example";
  const host = "corrupt.example";
  const storage = new MemoryStorage({
    [`fsrcnnx_site:${encodeURIComponent(host)}`]: { mode: "upscale", policy: "auto" },
    [fieldKey(scope, "mode")]: { schemaVersion: SETTINGS_SCHEMA_VERSION, value: "bad" },
  });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, legacyHosts: [host],
    fields: ["mode", "policy"], sourceId: "repair-corrupt", now: clock(100),
  });

  assert.deepEqual(await store.ready, { policy: "auto" }, "corruption does not resurrect the legacy mode");
  assert.equal(storage.setCalls.length, 0);
  assert.equal(store.health().state, "error");
  assert.equal(store.health().errorOperation, "validation");
  assert.equal(store.health().scope, scope);
  assert.match(store.health().error, /Corrupt v2 settings record: mode/);
  assert.ok(store.health().error.length <= 240);

  await store.write({ mode: "off" });
  assert.deepEqual(Object.keys(storage.setCalls[0]).sort(), [fieldKey(scope, "mode"), schemaKey(scope)].sort());
  assert.deepEqual(store.snapshot(), { policy: "auto", mode: "off" });
  assert.equal(store.health().state, "ready", "an explicit valid write repairs the corrupt field");
  store.close();
});

test("external patches are emitted, own events are suppressed, and tombstones remove fields", async () => {
  const scope = "https://events.example";
  const storage = seededStorage(scope);
  const a = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "a", now: clock(10),
  });
  const b = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "b", now: clock(20),
  });
  await Promise.all([a.ready, b.ready]);
  const patches = [];
  const unsubscribe = a.subscribe((patch, meta) => patches.push([patch, meta.source]));

  await a.write({ mode: "passthrough" });
  assert.deepEqual(patches, []);
  await b.write({ mode: "upscale" });
  assert.deepEqual(patches, [[{ mode: "upscale" }, "external"]]);
  await b.write({ mode: undefined });
  assert.equal(Object.prototype.hasOwnProperty.call(patches[1][0], "mode"), true);
  assert.equal(patches[1][0].mode, undefined);
  assert.deepEqual(a.snapshot(), {});

  storage.onChanged.emit({
    [fieldKey(scope, "mode")]: { newValue: record("ignored", { source: "other", time: 999 }) },
  }, "sync");
  assert.equal(patches.length, 2, "events from other storage areas are ignored");
  unsubscribe();
  await b.write({ mode: "off" });
  assert.equal(patches.length, 2);
  a.close();
  b.close();
});

test("storage rejection retains intent, exposes bounded health, and recovers through flush", async () => {
  const scope = "https://recovery.example";
  const storage = seededStorage(scope);
  storage.planSet({ error: new Error(`boom:${"x".repeat(400)}`) });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "recover", now: clock(10),
  });
  await store.ready;

  await assert.rejects(store.write({ mode: "upscale" }), /boom/);
  assert.deepEqual(store.snapshot(), { mode: "upscale" }, "failed persistence does not erase local intent");
  assert.equal(store.health().state, "error");
  assert.equal(store.health().errorOperation, "writing");
  assert.equal(store.health().pending, 1);
  assert.equal(store.health().error.length, 240);

  await store.flush();
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.deepEqual(store.health(), {
    state: "ready", operation: null, errorOperation: null, pending: 0, error: null,
    schemaVersion: SETTINGS_SCHEMA_VERSION, scope, closed: false,
  });
  store.close();
});

test("a failed initial read recovers through sync and permits later durable writes", async () => {
  const scope = "https://initial-recovery.example";
  const storage = seededStorage(scope, {
    mode: record("upscale", { source: "seed", time: 20 }),
  });
  storage.planGet({ error: new Error("initial storage read failed") });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"],
    sourceId: "initial-recovery", now: clock(100),
  });

  await assert.rejects(store.ready, /initial storage read failed/);
  assert.equal(storage.getCalls.length, 1);
  assert.equal(store.health().state, "error");
  assert.equal(store.health().operation, null);
  assert.equal(store.health().errorOperation, "loading");
  const patches = [];
  store.subscribe((patch, meta) => patches.push([patch, meta.source]));
  storage.onChanged.emit({
    [fieldKey(scope, "mode")]: {
      newValue: record("passthrough", { source: "pre-retry-event", time: 20 }),
    },
  });

  const retryRead = deferred();
  storage.planGet({ gate: retryRead });
  const sync = store.sync();
  await settle();
  assert.equal(store.health().operation, "loading",
    "sync exposes that it is retrying the failed initialization read");
  retryRead.resolve();

  assert.deepEqual(await sync, { mode: "upscale" });
  assert.equal(storage.getCalls.length, 2, "sync issues a fresh read after ready rejected");
  assert.deepEqual(store.snapshot(), { mode: "upscale" });
  assert.deepEqual(patches, [[{ mode: "upscale" }, "sync"]],
    "recovered values are published to the runtime application subscriber");
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale",
    "a pre-retry queued event cannot overwrite the later physical snapshot");
  assert.equal(store.health().error, null);
  assert.equal(store.health().errorOperation, null);

  await store.write({ mode: "off" });
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "off");
  await store.close();
});

test("a failed synchronization retains its operation provenance until recovery", async () => {
  const scope = "https://sync-recovery.example";
  const storage = seededStorage(scope);
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"],
    sourceId: "sync-recovery", now: clock(10),
  });
  await store.ready;
  storage.planGet({ error: new Error("refresh read failed") });

  await assert.rejects(store.sync(), /refresh read failed/);
  assert.equal(store.health().state, "error");
  assert.equal(store.health().operation, null);
  assert.equal(store.health().errorOperation, "syncing");

  assert.deepEqual(await store.sync(), {});
  assert.equal(store.health().state, "ready");
  assert.equal(store.health().errorOperation, null);
  await store.close();
});

test("sync reads canonical field keys without writing and emits its changed patch", async () => {
  const scope = "https://bfcache.example";
  const storage = seededStorage(scope, {
    mode: record("off", { source: "seed", time: 1 }),
  });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "resumed", now: clock(10),
  });
  await store.ready;
  const patches = [];
  store.subscribe((patch, meta) => patches.push([patch, meta.source]));
  storage.commit({
    [fieldKey(scope, "mode")]: record("upscale", { source: "other-tab", time: 20 }),
  }, { emit: false });
  const writesBefore = storage.setCalls.length;

  assert.deepEqual(await store.sync(), { mode: "upscale" });
  assert.deepEqual(store.snapshot(), { mode: "upscale" });
  assert.deepEqual(patches, [[{ mode: "upscale" }, "sync"]]);
  assert.equal(storage.setCalls.length, writesBefore, "sync performs no repair or migration writes");

  storage.commit({
    [fieldKey(scope, "mode")]: record("passthrough", { source: "aaa-equal-order", time: 20 }),
  }, { emit: false });
  assert.deepEqual(await store.sync(), { mode: "passthrough" },
    "the physically committed equal-order record wins without a source-name tie-break");
  assert.deepEqual(store.snapshot(), { mode: "passthrough" });
  assert.equal(storage.setCalls.length, writesBefore);
  store.close();
});

test("sync rejects a lower-stamp physical record and repairs canonical storage", async () => {
  const scope = "https://stale-physical.example";
  const canonical = record("upscale", { source: "observed", time: 50 });
  const storage = seededStorage(scope, { mode: canonical });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"],
    sourceId: "resumed", now: clock(100),
  });
  await store.ready;
  const patches = [];
  store.subscribe((patch, meta) => patches.push([patch, meta.source]));

  storage.commit({
    [fieldKey(scope, "mode")]: record("off", { source: "delayed-older", time: 10 }),
  }, { emit: false });

  assert.deepEqual(await store.sync(), {});
  assert.deepEqual(store.snapshot(), { mode: "upscale" },
    "a physically newer observation cannot reverse logical record order");
  assert.deepEqual(storage.data[fieldKey(scope, "mode")], canonical,
    "sync rewrites the cached canonical envelope over stale physical storage");
  assert.deepEqual(patches, [], "repairing storage does not emit a preference change");
  assert.equal(storage.setCalls.length, 1);
  await store.close();
});

test("sync rejects a stale snapshot after its in-flight write finishes", async () => {
  const scope = "https://sync-in-flight.example";
  const storage = seededStorage(scope, {
    mode: record("off", { source: "seed", time: 1 }),
  });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"],
    sourceId: "local", now: clock(100),
  });
  await store.ready;
  const delayedWrite = deferred();
  const staleRead = deferred();
  storage.planSet({ gate: delayedWrite });

  const write = store.write({ mode: "upscale" });
  await settle();
  storage.planGet({ gate: staleRead });
  const sync = store.sync();
  await settle();
  assert.equal(store.health().operation, "syncing",
    "health distinguishes an active storage refresh from a write");

  delayedWrite.resolve();
  await write;
  assert.equal(store.health().pending, 0,
    "the local-intent guard is gone before the stale read resolves");
  staleRead.resolve();

  assert.deepEqual(await sync, {});
  assert.deepEqual(store.snapshot(), { mode: "upscale" });
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.equal(storage.setCalls.length, 2,
    "the stale snapshot is repaired even though the original write already committed");
  await store.close();
});

test("a delayed sync cannot overwrite local intent created after its read began", async () => {
  const scope = "https://sync-race.example";
  const storage = seededStorage(scope, {
    mode: record("off", { source: "seed", time: 1 }),
  });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "local", now: clock(100),
  });
  await store.ready;
  const delayedRead = deferred();
  const delayedWrite = deferred();
  storage.planGet({ gate: delayedRead });
  storage.planSet({ gate: delayedWrite });

  const firstSync = store.sync();
  const secondSync = store.sync();
  assert.equal(firstSync, secondSync, "concurrent BFCache resyncs are coalesced");
  await settle();
  const localWrite = store.write({ mode: "upscale" });
  await settle();
  delayedRead.resolve();
  assert.deepEqual(await firstSync, {}, "the stale read is fenced by the later local generation");
  assert.deepEqual(store.snapshot(), { mode: "upscale" });

  delayedWrite.resolve();
  await localWrite;
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.equal(storage.setCalls.length, 1, "sync never adds a write to the local intent");
  store.close();
});

test("close removes observation and rejects future activity", async () => {
  const scope = "https://closed.example";
  const storage = seededStorage(scope);
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"], sourceId: "closed", now: clock(10),
  });
  await store.ready;
  let notifications = 0;
  store.subscribe(() => notifications++);
  assert.equal(storage.onChanged.listeners.size, 1);

  const firstClose = store.close();
  const secondClose = store.close();
  assert.equal(firstClose, secondClose, "close is idempotent while its promise is pending");
  await firstClose;
  assert.equal(storage.onChanged.listeners.size, 0);
  assert.equal(store.health().closed, true);
  storage.commit({
    [fieldKey(scope, "mode")]: record("upscale", { source: "other", time: 20 }),
  });
  assert.equal(notifications, 0);
  assert.deepEqual(store.snapshot(), {});
  await assert.rejects(store.write({ mode: "off" }), /closed/);
  await assert.rejects(store.sync(), /closed/);
});

test("close drains writes accepted behind an in-flight batch before resolving", async () => {
  const scope = "https://closing-write.example";
  const storage = seededStorage(scope);
  const firstCommit = deferred();
  storage.planSet({ gate: firstCommit });
  const store = createSettingsStore({
    storage, onChanged: storage.onChanged, scope, fields: ["mode"],
    sourceId: "closing", now: clock(10),
  });
  await store.ready;

  const firstWrite = store.write({ mode: "passthrough" });
  await settle();
  const queuedWrite = store.write({ mode: "upscale" });
  await settle();
  assert.equal(storage.setCalls.length, 1);
  assert.equal(store.health().pending, 1);

  let closeSettled = false;
  const closing = store.close().then(() => { closeSettled = true; });
  assert.equal(store.health().closed, true);
  assert.equal(storage.onChanged.listeners.size, 0,
    "close stops external observation immediately");
  await settle();
  assert.equal(closeSettled, false, "close waits for the active storage call");

  firstCommit.resolve();
  await Promise.all([firstWrite, queuedWrite, closing]);
  assert.equal(storage.setCalls.length, 2,
    "the queued accepted value drains after the active batch");
  assert.equal(storage.data[fieldKey(scope, "mode")].value, "upscale");
  assert.equal(store.health().pending, 0);
  await assert.rejects(store.write({ mode: "off" }), /closed/);
});
