// Per-scope settings persistence with field-level writes and convergent ordering.
// The module has no Chrome globals: callers inject a storage area and an
// onChanged-compatible event so the same implementation is executable in tests.

export const SETTINGS_SCHEMA_VERSION = 2;

export const DEFAULT_SETTING_FIELDS = Object.freeze([
  "mode",
  "engine",
  "artVariant",
  "policy",
  "ssimds",
  "sharpen",
  "sharpenStrength",
  "hoverReveal",
  "allVideos",
  "idlePowerSaving",
  "autoQualityFallback",
  "images",
  "interpolate",
  "interpEngine",
  "interpResMode",
  "neuralModel",
  "interpTargetFps",
  "interpAvOffsetMs",
  "interpStaticPassthrough",
  "interpAutoFallback",
  "interpLadder",
  "interpInvert",
]);

const MAX_ERROR_LENGTH = 240;
const LEGACY_MAP_KEY = "fsrcnnx_sites";
const SETTINGS_RECORD_MIGRATIONS = Object.freeze({
  1: (record) => ({ ...record, schemaVersion: 2 }),
});
const hasOwn = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function boundedError(error) {
  const message = error && typeof error.message === "string"
    ? error.message
    : String(error || "Unknown storage error");
  return message.slice(0, MAX_ERROR_LENGTH);
}

function compareRecords(left, right) {
  if (left.stamp.time !== right.stamp.time) return left.stamp.time - right.stamp.time;
  return left.stamp.counter - right.stamp.counter;
}

function sameValue(left, right) {
  return Object.is(left, right);
}

/**
 * Create a field-addressed settings store.
 *
 * Required options:
 * - storage: Promise-based { get(keys), set(items) } storage-area adapter.
 * - scope: canonical scope for new data (normally an exact origin).
 * - sourceId: unique identifier for this live document/context.
 *
 * Optional options:
 * - onChanged: Chrome-style { addListener, removeListener } event adapter.
 * - legacyHosts: hostname keys to read through from legacy data, highest precedence first.
 * - fields, now, areaName: schema fields, clock injection, and storage area.
 *
 * Record order uses a per-context hybrid clock, never a shared revision or CAS.
 * Strictly newer logical stamps win. Equal stamps from different contexts are
 * concurrent and resolve in observed commit-event order; source identifiers
 * identify echoes but never order user intent.
 *
 * Returns { ready, snapshot, write, flush, sync, subscribe, health, close }.
 * close() immediately rejects new activity and returns an idempotent promise
 * that settles after every write accepted before closure has drained.
 */
export function createSettingsStore({
  storage,
  onChanged = null,
  scope,
  sourceId,
  legacyHosts = [],
  fields = DEFAULT_SETTING_FIELDS,
  now = Date.now,
  areaName = "local",
} = {}) {
  if (!storage || typeof storage.get !== "function" || typeof storage.set !== "function") {
    throw new TypeError("storage must provide get() and set()");
  }
  if (typeof scope !== "string" || !scope) throw new TypeError("scope must be a non-empty string");
  if (typeof sourceId !== "string" || !sourceId || sourceId.length > 128 || sourceId.includes("\0")) {
    throw new TypeError("sourceId must be a non-empty string of at most 128 characters");
  }
  if (!Array.isArray(fields) || fields.length === 0) {
    throw new TypeError("fields must be a non-empty array");
  }
  if (!Array.isArray(legacyHosts) || legacyHosts.some((host) => typeof host !== "string" || !host)) {
    throw new TypeError("legacyHosts must contain only non-empty strings");
  }
  if (typeof now !== "function") throw new TypeError("now must be a function");

  const uniqueFields = [...new Set(fields)];
  if (uniqueFields.length !== fields.length ||
      uniqueFields.some((field) => typeof field !== "string" || !field || field === "$schema")) {
    throw new TypeError("fields must contain unique, non-empty, non-reserved strings");
  }

  const fieldSet = new Set(uniqueFields);
  const prefix = `fsrcnnx_setting:${encodeURIComponent(scope)}:`;
  const schemaKey = `${prefix}$schema`;
  const keyByField = new Map(uniqueFields.map((field) => [field, `${prefix}${encodeURIComponent(field)}`]));
  const fieldByKey = new Map([...keyByField].map(([field, key]) => [key, field]));
  const legacySiteKeys = legacyHosts.map((host) => `fsrcnnx_site:${encodeURIComponent(host)}`);

  const records = new Map();
  const values = new Map();
  const legacyFallbacks = new Map();
  const corruptFields = new Set();
  const supersededOwnRecords = new Set();
  const generations = new Map(uniqueFields.map((field) => [field, 0]));
  const subscribers = new Set();
  const queuedChanges = [];
  let pending = new Map(); // field -> { record, local }
  let inFlight = new Map();
  let pendingSchema = null;
  let schemaInFlight = false;
  let drainPromise = null;
  let syncPromise = null;
  let initializationPromise = null;
  let initialized = false;
  let initializing = false;
  let closed = false;
  let closePromise = null;
  let storageError = null;
  let storageErrorOperation = null;
  let corruptionError = null;
  let schemaPersisted = false;
  let clockTime = 0;
  let clockCounter = 0;

  function observeStamp(stamp) {
    if (stamp.time > clockTime) {
      clockTime = stamp.time;
      clockCounter = stamp.counter;
    } else if (stamp.time === clockTime) {
      clockCounter = Math.max(clockCounter, stamp.counter);
    }
  }

  function nextStamp() {
    const physical = Number(now());
    const usable = Number.isFinite(physical) ? Math.max(0, Math.trunc(physical)) : 0;
    if (usable > clockTime) {
      clockTime = usable;
      clockCounter = 0;
    } else {
      clockCounter++;
    }
    return { time: clockTime, counter: clockCounter };
  }

  function makeRecord(value) {
    const record = {
      schemaVersion: SETTINGS_SCHEMA_VERSION,
      source: sourceId,
      stamp: nextStamp(),
    };
    if (value === undefined) record.deleted = true;
    else record.value = value;
    return record;
  }

  function migrateRecordEnvelope(value) {
    if (!isObject(value) || !Number.isSafeInteger(value.schemaVersion) ||
        value.schemaVersion < 1) return null;
    let migrated = value;
    while (migrated.schemaVersion < SETTINGS_SCHEMA_VERSION) {
      const previousVersion = migrated.schemaVersion;
      const migrate = SETTINGS_RECORD_MIGRATIONS[previousVersion];
      if (typeof migrate !== "function") return null;
      try { migrated = migrate(migrated); }
      catch { return null; }
      if (!isObject(migrated) || migrated.schemaVersion !== previousVersion + 1) return null;
    }
    return migrated;
  }

  function parseRecord(value) {
    const migrated = migrateRecordEnvelope(value);
    if (!migrated || typeof migrated.source !== "string" || !migrated.source ||
        migrated.source.length > 128 || migrated.source.includes("\0") ||
        !isObject(migrated.stamp) ||
        !Number.isSafeInteger(migrated.stamp.time) || migrated.stamp.time < 0 ||
        !Number.isSafeInteger(migrated.stamp.counter) || migrated.stamp.counter < 0) return null;
    const deleted = migrated.deleted === true;
    if (deleted === hasOwn(migrated, "value")) return null;
    return deleted
      ? { schemaVersion: migrated.schemaVersion, source: migrated.source,
          stamp: { time: migrated.stamp.time, counter: migrated.stamp.counter }, deleted: true }
      : { schemaVersion: migrated.schemaVersion, source: migrated.source,
          stamp: { time: migrated.stamp.time, counter: migrated.stamp.counter }, value: migrated.value };
  }

  function hasCompatibleSchemaMarker(value) {
    return isObject(value) && Number.isSafeInteger(value.schemaVersion) &&
      value.schemaVersion >= SETTINGS_SCHEMA_VERSION;
  }

  function recordValue(record) {
    return record?.deleted ? undefined : record?.value;
  }

  function recordToken(record) {
    return `${record.source}\u0000${record.stamp.time}\u0000${record.stamp.counter}`;
  }

  function refreshCorruptionError() {
    corruptionError = corruptFields.size
      ? boundedError(`Invalid settings record: ${[...corruptFields].sort().join(", ")}`)
      : null;
  }

  function setCachedRecord(field, record) {
    records.set(field, record);
    corruptFields.delete(field);
    refreshCorruptionError();
    if (record.deleted) values.delete(field);
    else values.set(field, record.value);
    generations.set(field, generations.get(field) + 1);
  }

  function setLegacyFallback(field) {
    const previous = values.get(field);
    const hadPrevious = values.has(field);
    records.delete(field);
    corruptFields.delete(field);
    refreshCorruptionError();
    if (legacyFallbacks.has(field)) values.set(field, legacyFallbacks.get(field));
    else values.delete(field);
    const hasNext = values.has(field);
    const next = values.get(field);
    const changed = hadPrevious !== hasNext || !sameValue(previous, next);
    if (changed) generations.set(field, generations.get(field) + 1);
    return { changed, value: next };
  }

  function setCorruptField(field) {
    const previous = values.get(field);
    const hadPrevious = values.has(field);
    records.delete(field);
    values.delete(field);
    corruptFields.add(field);
    refreshCorruptionError();
    if (hadPrevious) generations.set(field, generations.get(field) + 1);
    return hadPrevious;
  }

  function snapshot() {
    return Object.fromEntries(values);
  }

  function pendingCount() {
    const fieldNames = new Set([...pending.keys(), ...inFlight.keys()]);
    return fieldNames.size;
  }

  function health() {
    const operation = initializing ? "loading"
      : syncPromise !== null ? "syncing"
      : drainPromise !== null ? "writing"
      : null;
    const active = operation !== null;
    const error = storageError || corruptionError;
    return {
      state: active ? "writing" : error ? "error" : "ready",
      operation,
      errorOperation: storageError ? storageErrorOperation : corruptionError ? "validation" : null,
      pending: pendingCount(),
      error,
      schemaVersion: SETTINGS_SCHEMA_VERSION,
      scope,
      closed,
    };
  }

  function emitPatch(patch, source) {
    if (Object.keys(patch).length === 0) return;
    for (const listener of [...subscribers]) {
      try { listener({ ...patch }, { source, scope }); }
      catch {}
    }
  }

  function newerEntry(left, right) {
    if (!left) return right;
    if (!right) return left;
    const ordering = compareRecords(left.record, right.record);
    if (ordering !== 0) return ordering > 0 ? left : right;
    return right.local && !left.local ? right : left;
  }

  function latestLocalIntent(field) {
    const queued = pending.get(field);
    const active = inFlight.get(field);
    return newerEntry(queued?.local ? queued : null, active?.local ? active : null)?.record || null;
  }

  function queueCanonicalRepair(field, canonical) {
    if (closed || !canonical) return null;
    // Rewriting the exact canonical envelope converges physical storage without
    // inventing a newer user intent. Equal repair events are then inert in every
    // context, which prevents source-to-source repair ping-pong.
    const repair = { record: canonical, local: false };
    pending.set(field, newerEntry(pending.get(field), repair));
    const repairDrain = startDrain();
    void repairDrain.catch(() => {});
    return repairDrain;
  }

  function acceptExternalRecord(field, incoming, patch) {
    const hadPrevious = values.has(field);
    const previousValue = values.get(field);
    const incomingValue = recordValue(incoming);
    setCachedRecord(field, incoming);
    const hasIncoming = !incoming.deleted;
    if (hadPrevious !== hasIncoming || !sameValue(previousValue, incomingValue)) {
      patch[field] = incomingValue;
    }
  }

  function rememberSuperseded(record) {
    supersededOwnRecords.add(recordToken(record));
    if (supersededOwnRecords.size > 256) {
      supersededOwnRecords.delete(supersededOwnRecords.values().next().value);
    }
  }

  function supersedeLocalAtOrBefore(field, incoming) {
    const queued = pending.get(field);
    if (queued?.local && compareRecords(queued.record, incoming) <= 0) pending.delete(field);
    const active = inFlight.get(field);
    if (active?.local && compareRecords(active.record, incoming) <= 0) rememberSuperseded(active.record);
  }

  function applyChangeBatch(changes, { emit = true, repair = true } = {}) {
    if (!isObject(changes)) return {};
    const schemaChange = changes[schemaKey];
    if (isObject(schemaChange) && hasOwn(schemaChange, "newValue")) {
      schemaPersisted = hasCompatibleSchemaMarker(schemaChange.newValue);
    }
    const patch = {};
    for (const [key, change] of Object.entries(changes)) {
      const field = fieldByKey.get(key);
      if (!field || !isObject(change)) continue;
      const current = records.get(field);
      if (!hasOwn(change, "newValue") || change.newValue === undefined) {
        if (latestLocalIntent(field)) {
          if (repair) queueCanonicalRepair(field, current);
          continue;
        }
        const fallback = setLegacyFallback(field);
        if (fallback.changed) patch[field] = fallback.value;
        continue;
      }

      const incoming = parseRecord(change.newValue);
      if (!incoming) {
        if (latestLocalIntent(field)) {
          if (repair) queueCanonicalRepair(field, current);
          continue;
        }
        if (setCorruptField(field)) patch[field] = undefined;
        continue;
      }
      observeStamp(incoming.stamp);

      if (incoming.source === sourceId) {
        const token = recordToken(incoming);
        if (supersededOwnRecords.has(token)) {
          supersededOwnRecords.delete(token);
          if (current && repair) queueCanonicalRepair(field, current);
          continue;
        }
        if (!current) {
          setCachedRecord(field, incoming);
          continue;
        }
        const ordering = compareRecords(incoming, current);
        if (ordering < 0 && repair) {
          queueCanonicalRepair(field, current);
        } else if (ordering > 0) {
          setCachedRecord(field, incoming);
        } else if (incoming.source !== current.source) {
          // Equal cross-context stamps are concurrent. This observed own commit
          // is the later commit unless it was explicitly superseded above.
          acceptExternalRecord(field, incoming, patch);
        }
        continue;
      }

      const localIntent = latestLocalIntent(field);
      if (localIntent) {
        const ordering = compareRecords(localIntent, incoming);
        if (ordering > 0) {
          if (repair) queueCanonicalRepair(field, records.get(field));
          continue;
        }
        // Equal stamps from different sources are never ordered by source ID.
        // The incoming change is the later observed commit; an older in-flight
        // own write is marked so its eventual echo repairs this canonical value.
        supersedeLocalAtOrBefore(field, incoming);
      }

      if (current) {
        const ordering = compareRecords(incoming, current);
        if (ordering < 0) {
          if (repair) queueCanonicalRepair(field, current);
          continue;
        }
        if (ordering === 0 && incoming.source === current.source) continue;
      }
      acceptExternalRecord(field, incoming, patch);
    }
    if (emit) emitPatch(patch, "external");
    return patch;
  }

  function changeListener(changes, changedArea) {
    if (closed || (changedArea != null && changedArea !== areaName)) return;
    if (!initialized) queuedChanges.push(changes);
    else applyChangeBatch(changes);
  }

  if (onChanged != null) {
    if (typeof onChanged.addListener !== "function" || typeof onChanged.removeListener !== "function") {
      throw new TypeError("onChanged must provide addListener() and removeListener()");
    }
    onChanged.addListener(changeListener);
  }

  function mergeFailedBatch(batch) {
    for (const [field, entry] of batch) {
      if (entry.local && supersededOwnRecords.has(recordToken(entry.record))) continue;
      pending.set(field, newerEntry(pending.get(field), entry));
    }
  }

  async function drain() {
    // Closing prevents new activity, but writes accepted before close() must
    // remain durable. Continue through every batch already queued behind the
    // active storage call before allowing the close promise to settle.
    while (pending.size > 0 || pendingSchema) {
      const batch = pending;
      const schema = pendingSchema;
      pending = new Map();
      pendingSchema = null;
      inFlight = batch;
      schemaInFlight = !!schema;
      const items = {};
      for (const [field, entry] of batch) items[keyByField.get(field)] = entry.record;
      if (schema) items[schemaKey] = schema;
      try {
        await storage.set(items);
        if (schema) schemaPersisted = true;
      } catch (error) {
        mergeFailedBatch(batch);
        if (schema && !pendingSchema) pendingSchema = schema;
        storageError = boundedError(error);
        storageErrorOperation = "writing";
        throw error;
      } finally {
        inFlight = new Map();
        schemaInFlight = false;
      }
    }
    if (pendingCount() === 0) {
      storageError = null;
      storageErrorOperation = null;
    }
  }

  function ensureDrain() {
    if (!drainPromise) {
      const active = drain();
      const wrapped = active.finally(() => {
        if (drainPromise === wrapped) drainPromise = null;
      });
      drainPromise = wrapped;
    }
    return drainPromise;
  }

  function startDrain() {
    if (closed) return Promise.reject(new Error("settings store is closed"));
    return ensureDrain();
  }

  function legacyValues(stored) {
    const migrated = Object.create(null);
    const oldMap = isObject(stored[LEGACY_MAP_KEY]) ? stored[LEGACY_MAP_KEY] : null;
    // The first legacy hostname is the canonical predecessor. Iterate in
    // reverse so it has highest precedence when aliases were supplied.
    for (const host of [...legacyHosts].reverse()) {
      const candidate = oldMap && hasOwn(oldMap, host) && isObject(oldMap[host]) ? oldMap[host] : null;
      if (candidate) Object.assign(migrated, candidate);
    }
    for (const key of [...legacySiteKeys].reverse()) {
      const candidate = hasOwn(stored, key) && isObject(stored[key]) ? stored[key] : null;
      if (candidate) Object.assign(migrated, candidate);
    }
    return migrated;
  }

  async function initialize() {
    // Events already queued before this read are represented by the newer
    // physical snapshot. Only changes delivered while the read is in flight
    // need to be replayed over it. Preserve the whole queue when the read
    // fails so the next successful snapshot can establish that boundary.
    const queuedBeforeRead = queuedChanges.length;
    const stored = await storage.get([
      schemaKey,
      ...keyByField.values(),
      ...legacySiteKeys,
      LEGACY_MAP_KEY,
    ]);
    if (!isObject(stored)) throw new TypeError("storage.get() must resolve to an object");

    const legacy = legacyValues(stored);
    for (const field of uniqueFields) {
      if (hasOwn(legacy, field) && legacy[field] !== undefined) legacyFallbacks.set(field, legacy[field]);
    }

    const marker = stored[schemaKey];
    schemaPersisted = hasCompatibleSchemaMarker(marker);
    for (const field of uniqueFields) {
      const key = keyByField.get(field);
      if (!hasOwn(stored, key)) {
        if (legacyFallbacks.has(field)) {
          values.set(field, legacyFallbacks.get(field));
          generations.set(field, generations.get(field) + 1);
        }
        continue;
      }
      const record = parseRecord(stored[key]);
      if (!record) {
        setCorruptField(field);
        continue;
      }
      observeStamp(record.stamp);
      setCachedRecord(field, record);
    }

    // Legacy hostname-keyed records were read forever and never removed, so an
    // upgraded user retained an indefinite record of every site they had ever
    // configured, in addition to the current origin-keyed records. They are only
    // consulted as a fallback when a field has no current record, so they are safe
    // to delete once every field they supplied has one — which makes this a no-op
    // on fresh installs and a one-time cleanup for migrated ones. Deleting earlier
    // would drop values the fallback is still serving.
    const legacyKeysPresent = [...legacySiteKeys, LEGACY_MAP_KEY].filter((key) => hasOwn(stored, key));
    if (legacyKeysPresent.length && typeof storage.remove === "function") {
      const stillNeeded = [...legacyFallbacks.keys()].some((field) => !records.has(field));
      if (!stillNeeded) {
        try {
          await storage.remove(legacyKeysPresent);
          legacyFallbacks.clear();
        } catch {
          // A failed cleanup must never block initialization; it retries next load.
        }
      }
    }

    initialized = true;
    storageError = null;
    storageErrorOperation = null;
    if (queuedBeforeRead > 0) queuedChanges.splice(0, queuedBeforeRead);
    for (const changes of queuedChanges.splice(0)) applyChangeBatch(changes);
    return snapshot();
  }

  function ensureInitialized() {
    if (closed) return Promise.reject(new Error("settings store is closed"));
    if (initialized) return Promise.resolve(snapshot());
    if (!initializationPromise) {
      initializing = true;
      // A later lifecycle sync is the recovery path for a transient initial
      // read failure. Publish the retry as active instead of retaining a stale
      // terminal error while storage.get() is in flight again.
      storageError = null;
      storageErrorOperation = null;
      const active = initialize();
      const wrapped = active.catch((error) => {
        storageError = boundedError(error);
        storageErrorOperation = "loading";
        throw error;
      }).finally(() => {
        if (initializationPromise === wrapped) initializationPromise = null;
        initializing = false;
      });
      initializationPromise = wrapped;
    }
    return initializationPromise;
  }

  const ready = ensureInitialized();

  async function write(patch) {
    await ensureInitialized();
    if (closed) throw new Error("settings store is closed");
    if (!isObject(patch)) throw new TypeError("patch must be an object");
    const entries = Object.entries(patch);
    for (const [field] of entries) {
      if (!fieldSet.has(field)) throw new TypeError(`unknown settings field: ${field}`);
    }
    for (const [field, value] of entries) {
      const record = makeRecord(value);
      setCachedRecord(field, record);
      pending.set(field, { record, local: true });
    }
    if (entries.length > 0) {
      if (!schemaPersisted && !schemaInFlight && !pendingSchema) {
        pendingSchema = {
          schemaVersion: SETTINGS_SCHEMA_VERSION,
          source: sourceId,
          establishedAt: Math.max(0, Math.trunc(Number(now()) || 0)),
        };
      }
      await startDrain();
    }
    return snapshot();
  }

  async function flush() {
    await ensureInitialized();
    if (closed) throw new Error("settings store is closed");
    if (drainPromise || pending.size > 0 || pendingSchema) await startDrain();
    return snapshot();
  }

  async function performSync() {
    const wasInitialized = initialized;
    await ensureInitialized();
    if (closed) throw new Error("settings store is closed");
    // Retrying initialization already consumed an authoritative full snapshot.
    // Publish that recovered snapshot through the normal sync channel so a
    // caller whose initial restore failed can apply durable values instead of
    // remaining on in-memory defaults. Avoid a redundant second read.
    if (!wasInitialized) {
      const recovered = snapshot();
      emitPatch(recovered, "sync");
      return recovered;
    }
    const startingGenerations = new Map(generations);
    let stored;
    try {
      stored = await storage.get([...keyByField.values()]);
    } catch (error) {
      storageError = boundedError(error);
      storageErrorOperation = "syncing";
      throw error;
    }
    if (!isObject(stored)) {
      const error = new TypeError("storage.get() must resolve to an object");
      storageError = boundedError(error);
      storageErrorOperation = "syncing";
      throw error;
    }

    const patch = {};
    const repairDrains = new Set();
    for (const field of uniqueFields) {
      // A local write or observed event that happened after this read began is
      // more authoritative than the potentially stale read result.
      if (generations.get(field) !== startingGenerations.get(field) || latestLocalIntent(field)) continue;
      const current = records.get(field);
      const key = keyByField.get(field);
      if (!hasOwn(stored, key)) {
        const fallback = setLegacyFallback(field);
        if (fallback.changed) patch[field] = fallback.value;
        continue;
      }
      const next = parseRecord(stored[key]);
      if (!next) {
        if (setCorruptField(field)) patch[field] = undefined;
        continue;
      }
      observeStamp(next.stamp);
      if (current) {
        const ordering = compareRecords(next, current);
        if (ordering < 0) {
          const repairDrain = queueCanonicalRepair(field, current);
          if (repairDrain) repairDrains.add(repairDrain);
          continue;
        }
        if (ordering === 0 && current.source === next.source) continue;
      }
      const hadPrevious = values.has(field);
      const previousValue = values.get(field);
      const nextValue = recordValue(next);
      setCachedRecord(field, next);
      const hasNext = !next.deleted;
      if (hadPrevious !== hasNext || !sameValue(previousValue, nextValue)) patch[field] = nextValue;
    }
    emitPatch(patch, "sync");
    if (repairDrains.size > 0) await Promise.all(repairDrains);
    if (pendingCount() === 0) {
      storageError = null;
      storageErrorOperation = null;
    }
    return patch;
  }

  function sync() {
    if (closed) return Promise.reject(new Error("settings store is closed"));
    if (!syncPromise) {
      const active = performSync();
      const wrapped = active.finally(() => {
        if (syncPromise === wrapped) syncPromise = null;
      });
      syncPromise = wrapped;
    }
    return syncPromise;
  }

  function subscribe(listener) {
    if (typeof listener !== "function") throw new TypeError("listener must be a function");
    if (closed) throw new Error("settings store is closed");
    subscribers.add(listener);
    return () => subscribers.delete(listener);
  }

  function close() {
    if (closePromise) return closePromise;
    closed = true;
    subscribers.clear();
    if (onChanged) onChanged.removeListener(changeListener);
    closePromise = drainPromise || (pending.size > 0 || pendingSchema
      ? ensureDrain()
      : Promise.resolve());
    return closePromise;
  }

  return Object.freeze({ ready, snapshot, write, flush, sync, subscribe, health, close });
}
