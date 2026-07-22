import test from "node:test";
import assert from "node:assert/strict";

import * as rife from "../fsrcnnx-rife.js";

let runtimeRevision = 0;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail(message);
}

function installRuntimeMocks(t, state) {
  const chromeDescriptor = Object.getOwnPropertyDescriptor(globalThis, "chrome");
  const fetchDescriptor = Object.getOwnPropertyDescriptor(globalThis, "fetch");
  const stateDescriptor = Object.getOwnPropertyDescriptor(globalThis, "__rifeSessionLifecycle");
  const runtimeSource = `
    // Runtime fixture ${++runtimeRevision}
    const state = globalThis.__rifeSessionLifecycle;
    export const env = state.env;
    export class Tensor {
      constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
    }
    export class InferenceSession {
      static create(url, options) { return state.create(url, options); }
    }
  `;
  const runtimeUrl = `data:text/javascript,${encodeURIComponent(runtimeSource)}`;

  globalThis.__rifeSessionLifecycle = state;
  globalThis.chrome = {
    runtime: {
      getURL(path) {
        if (path === "vendor/ort/ort.webgpu.min.mjs") return runtimeUrl;
        return `https://extension.test/${path}`;
      },
    },
  };
  globalThis.fetch = async () => ({ ok: true, status: 200 });

  t.after(() => {
    if (chromeDescriptor) Object.defineProperty(globalThis, "chrome", chromeDescriptor);
    else delete globalThis.chrome;
    if (fetchDescriptor) Object.defineProperty(globalThis, "fetch", fetchDescriptor);
    else delete globalThis.fetch;
    if (stateDescriptor) Object.defineProperty(globalThis, "__rifeSessionLifecycle", stateDescriptor);
    else delete globalThis.__rifeSessionLifecycle;
  });
}

test("RIFE initialization rejects an already-lost ORT device", async (t) => {
  const lostDevice = {
    id: "already-lost",
    lost: Promise.resolve({ reason: "unknown", message: "already gone" }),
  };
  let releases = 0;
  const session = {
    inputNames: ["input"],
    outputNames: ["output"],
    async release() { releases++; },
  };
  const state = {
    env: { wasm: {}, webgpu: { enableFp16: false, device: lostDevice } },
    create() {
      state.env.webgpu.device = lostDevice;
      return session;
    },
  };
  installRuntimeMocks(t, state);
  const isolated = await import(`../fsrcnnx-rife.js?already-lost=${Date.now()}`);

  assert.equal(await isolated.initRife(), false);
  assert.equal(isolated.isReady(), false);
  assert.equal(isolated.getOrtDevice(), null);
  assert.equal(releases, 1, "the rejected committed session is released exactly once");
  await isolated.disposeRife();
});

test("RIFE session replacement, loss invalidation, and disposal preserve physical ownership", async (t) => {
  const events = [];
  const requests = [];
  const initialLoss = deferred();
  const recoveredLoss = deferred();
  const devices = {
    initial: { id: "device-initial", lost: initialLoss.promise },
    stale: { id: "device-stale" },
    latest: { id: "device-latest" },
    recovered: { id: "device-recovered", lost: recoveredLoss.promise },
    invalidated: { id: "device-invalidated", lost: new Promise(() => {}) },
    disposal: { id: "device-disposal", lost: new Promise(() => {}) },
    resumed: { id: "device-resumed", lost: new Promise(() => {}) },
    guardLatest: { id: "device-guard-latest", lost: new Promise(() => {}) },
  };
  const releases = {
    initial: 0, stale: 0, latest: 0, same: 0, recovered: 0, invalidated: 0,
    disposal: 0, resumed: 0, guardLatest: 0,
  };

  const makeSession = (id, device) => ({
    id,
    device,
    inputNames: [`input-${id}`],
    outputNames: [`output-${id}`],
    async release() {
      releases[id]++;
      events.push({
        type: `release-${id}`,
        activeDevice: rife.getOrtDevice(),
        ready: rife.isReady(),
      });
    },
  });
  const initialSession = makeSession("initial", devices.initial);
  const staleSession = makeSession("stale", devices.stale);
  const latestSession = makeSession("latest", devices.latest);
  const sameDeviceSession = makeSession("same", devices.latest);
  const recoveredSession = makeSession("recovered", devices.recovered);
  const invalidationReleaseStarted = deferred();
  const invalidationRelease = deferred();
  let invalidationReleaseInProgress = false;
  let createdDuringInvalidationRelease = false;
  const invalidatedSession = {
    ...makeSession("invalidated", devices.invalidated),
    async release() {
      releases.invalidated++;
      invalidationReleaseInProgress = true;
      invalidationReleaseStarted.resolve();
      await invalidationRelease.promise;
      invalidationReleaseInProgress = false;
    },
  };
  const disposalRelease = deferred();
  const disposalSession = {
    ...makeSession("disposal", devices.disposal),
    async release() {
      releases.disposal++;
      await disposalRelease.promise;
    },
  };
  const confirmedGuardReleaseStarted = deferred();
  const confirmedGuardRelease = deferred();
  const resumedSession = {
    ...makeSession("resumed", devices.resumed),
    async release() {
      releases.resumed++;
      confirmedGuardReleaseStarted.resolve();
      await confirmedGuardRelease.promise;
    },
  };
  const guardLatestSession = makeSession("guardLatest", devices.guardLatest);
  let createCount = 0;
  const state = {
    env: { wasm: {}, webgpu: { enableFp16: false, device: null } },
    create(url) {
      createCount++;
      if (invalidationReleaseInProgress) createdDuringInvalidationRelease = true;
      if (createCount === 1) {
        state.env.webgpu.device = devices.initial;
        events.push({ type: "create-initial" });
        return initialSession;
      }
      const request = deferred();
      requests.push({
        url,
        resolve(session) {
          state.env.webgpu.device = session.device;
          events.push({ type: `create-${session.id}` });
          request.resolve(session);
        },
      });
      return request.promise;
    },
  };
  installRuntimeMocks(t, state);

  assert.equal(await rife.initRife(), true);
  assert.equal(rife.getOrtDevice(), devices.initial);
  assert.equal(rife.isReady(), true);

  assert.equal(rife.setModel("rife_v4.26_fp16"), true);
  const staleInit = rife.initRife();
  await waitFor(() => requests.length === 1, "stale candidate creation did not start");
  assert.equal(releases.initial, 0);

  assert.equal(rife.setModel("rife_v4.26"), true);
  const latestInit = rife.initRife();
  requests[0].resolve(staleSession);
  assert.equal(await staleInit, false);
  assert.equal(releases.stale, 1);
  assert.equal(releases.initial, 0);
  const staleRelease = events.find(({ type }) => type === "release-stale");
  assert.equal(staleRelease.activeDevice, devices.initial);
  assert.equal(staleRelease.ready, false);

  await waitFor(() => requests.length === 2, "latest candidate creation did not start");
  assert.equal(releases.initial, 0);
  requests[1].resolve(latestSession);
  assert.equal(await latestInit, true);

  // The devices differ, so the initial session must remain a lifetime guard
  // until the renderer confirms it has adopted the replacement device.
  assert.equal(releases.initial, 0);
  assert.equal(releases.stale, 1);
  assert.equal(releases.latest, 0);
  assert.equal(rife.getOrtDevice(), devices.latest);
  assert.equal(rife.isReady(), true);
  assert.equal(rife.listModels().find(({ current }) => current)?.key, "rife_v4.26");

  const latestCreateIndex = events.findIndex(({ type }) => type === "create-latest");
  assert.equal(await rife.confirmOrtDeviceAdopted(devices.initial), false);
  assert.equal(releases.initial, 0);
  initialLoss.resolve({ message: "old adopted device lost" });
  await waitFor(() => releases.initial === 1,
    "natural loss of the non-current RIFE guard did not release its session");
  assert.equal(rife.getOrtDevice(), devices.latest,
    "old-guard loss must preserve the healthy replacement session");
  assert.equal(rife.isReady(), true);
  assert.equal(await rife.confirmOrtDeviceAdopted(devices.latest), true);
  assert.equal(releases.initial, 1);
  const initialReleaseIndex = events.findIndex(({ type }) => type === "release-initial");
  assert.ok(latestCreateIndex >= 0 && latestCreateIndex < initialReleaseIndex);
  const initialRelease = events[initialReleaseIndex];
  assert.equal(initialRelease.activeDevice, devices.latest);
  assert.equal(initialRelease.ready, true);

  // A subsequent replacement on the already-adopted device can release its
  // predecessor immediately because the candidate itself preserves that device.
  assert.equal(rife.setModel("rife_v4.26_fp16"), true);
  const sameDeviceInit = rife.initRife();
  await waitFor(() => requests.length === 3, "same-device candidate creation did not start");
  requests[2].resolve(sameDeviceSession);
  assert.equal(await sameDeviceInit, true);
  assert.equal(releases.latest, 1);
  assert.equal(releases.same, 0);
  const latestRelease = events.find(({ type }) => type === "release-latest");
  assert.equal(latestRelease.activeDevice, devices.latest);
  assert.equal(latestRelease.ready, true);

  // A lost committed device must invalidate the reusable session immediately.
  // The next init for the same selected model then creates a new session/device.
  const losses = [];
  const unsubscribe = rife.addDeviceLossListener((device, info) => losses.push({ device, info }));
  assert.equal(await rife.invalidateDevice(devices.latest, { message: "adapter reset" }), true);
  assert.equal(rife.isReady(), false);
  assert.equal(rife.getOrtDevice(), null);
  assert.equal(releases.same, 1);
  assert.equal(losses.length, 1);
  assert.equal(losses[0].device, devices.latest);

  const recoveredInit = rife.initRife();
  await waitFor(() => requests.length === 4, "post-loss candidate creation did not start");
  requests[3].resolve(recoveredSession);
  assert.equal(await recoveredInit, true);
  assert.equal(rife.isReady(), true);
  assert.equal(rife.getOrtDevice(), devices.recovered);
  assert.equal(rife.gpuActive(), false, "natural session loss must not depend on GpuInterp");

  const naturalInfo = { reason: "unknown", message: "natural adapter reset" };
  recoveredLoss.resolve(naturalInfo);
  await waitFor(() => !rife.isReady(), "natural ORT device loss did not invalidate the session");
  await waitFor(() => releases.recovered === 1, "naturally lost ORT session was not released");
  assert.equal(rife.getOrtDevice(), null);
  assert.equal(losses.length, 2);
  assert.deepEqual(losses[1], { device: devices.recovered, info: naturalInfo });
  unsubscribe();

  // Reinitialization and disposal must both wait for invalidation-owned physical
  // cleanup. Calling either while release() is blocked must not create a new ORT
  // session, resolve disposal, or double-release the unpublished lost session.
  const invalidatedInit = rife.initRife();
  await waitFor(() => requests.length === 5, "invalidation fixture session creation did not start");
  requests[4].resolve(invalidatedSession);
  assert.equal(await invalidatedInit, true);
  const invalidation = rife.invalidateDevice(devices.invalidated, { message: "deferred reset" });
  assert.equal(rife.isReady(), false, "loss must unpublish readiness synchronously");
  assert.equal(rife.getOrtDevice(), null, "loss must unpublish the device synchronously");
  await invalidationReleaseStarted.promise;

  const recoveryBehindInvalidation = rife.initRife();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(requests.length, 5,
    "recovery must not create a session while invalidated release() is pending");
  const disposalBehindInvalidation = rife.disposeRife();
  assert.equal(rife.disposeRife(), disposalBehindInvalidation,
    "disposal remains single-flight while invalidation owns the session");
  let disposalFinished = false;
  disposalBehindInvalidation.then(() => { disposalFinished = true; });
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(requests.length, 5);
  assert.equal(disposalFinished, false,
    "disposal must not resolve while invalidated release() is pending");
  assert.equal(releases.invalidated, 1);

  invalidationRelease.resolve();
  assert.equal(await invalidation, true);
  await disposalBehindInvalidation;
  assert.equal(disposalFinished, true);
  assert.equal(releases.invalidated, 1, "the invalidated session is released exactly once");
  await waitFor(() => requests.length === 6, "post-invalidation recovery did not start");
  requests[5].resolve(disposalSession);
  assert.equal(await recoveryBehindInvalidation, true);
  assert.equal(createdDuringInvalidationRelease, false);
  assert.equal(rife.getOrtDevice(), devices.disposal);

  // Intentional retirement makes public state idle immediately, waits for the
  // session release, and prevents a new init from overtaking that release.
  const retirement = rife.disposeRife();
  assert.equal(rife.isReady(), false);
  assert.equal(rife.getOrtDevice(), null);
  const resumedInit = rife.initRife();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(requests.length, 6, "reinitialization must wait for physical session release");
  assert.equal(releases.disposal, 1);
  disposalRelease.resolve();
  await retirement;
  await waitFor(() => requests.length === 7, "post-retirement session creation did not start");
  requests[6].resolve(resumedSession);
  assert.equal(await resumedInit, true);
  assert.equal(rife.getOrtDevice(), devices.resumed);

  // A confirmation claims its guard synchronously, so disposal's own snapshots
  // are empty while the serialized release tail is still pending. Disposal must
  // nevertheless wait for that physical release before releasing the current
  // replacement and resolving.
  assert.equal(rife.setModel("rife_v4.26"), true);
  const guardLatestInit = rife.initRife();
  await waitFor(() => requests.length === 8, "guard-tail replacement creation did not start");
  requests[7].resolve(guardLatestSession);
  assert.equal(await guardLatestInit, true);
  assert.equal(releases.resumed, 0);
  const confirmation = rife.confirmOrtDeviceAdopted(devices.guardLatest);
  await confirmedGuardReleaseStarted.promise;
  const finalDisposal = rife.disposeRife();
  let finalDisposalFinished = false;
  finalDisposal.then(() => { finalDisposalFinished = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(finalDisposalFinished, false,
    "disposal must await a guard release already claimed by confirmation");
  assert.equal(releases.guardLatest, 0,
    "the current session must remain alive behind the prior guard release tail");

  confirmedGuardRelease.resolve();
  assert.equal(await confirmation, true);
  await finalDisposal;
  assert.equal(releases.resumed, 1);
  assert.equal(releases.guardLatest, 1);
});

test("RIFE loss failures survive duplicate, unrelated, and settled recovery barriers", async (t) => {
  const loss = deferred();
  const releaseStarted = deferred();
  const releaseGate = deferred();
  const device = { id: "release-failure", lost: loss.promise };
  const unrelatedDevice = { id: "unrelated-loss", lost: new Promise(() => {}) };
  const physicalFailure = new Error("RIFE ORT release failed");
  let creates = 0;
  let releases = 0;
  const session = {
    inputNames: ["input"],
    outputNames: ["output"],
    async release() {
      releases++;
      releaseStarted.resolve();
      await releaseGate.promise;
      throw physicalFailure;
    },
  };
  const state = {
    env: { wasm: {}, webgpu: { enableFp16: false, device } },
    create() {
      creates++;
      state.env.webgpu.device = device;
      return session;
    },
  };
  installRuntimeMocks(t, state);
  const isolated = await import(
    `../fsrcnnx-rife.js?release-failure=${Date.now()}-${runtimeRevision}`
  );
  assert.equal(await isolated.initRife(), true);

  // The provider watcher claims first; the renderer then observes the same loss.
  loss.resolve({ message: "natural release failure" });
  await releaseStarted.promise;
  const duplicate = isolated.invalidateDevice(device, { message: "renderer duplicate" });
  assert.strictEqual(isolated.invalidateDevice(device), duplicate);
  const unrelated = isolated.invalidateDevice(unrelatedDevice);
  const recovery = isolated.initRife();
  const settled = Promise.allSettled([duplicate, unrelated, recovery]);

  releaseGate.resolve();
  const results = await settled;
  assert.equal(results[0].status, "rejected");
  assert.ok(results[0].reason instanceof AggregateError);
  assert.equal(results[1].status, "rejected");
  assert.ok(results[1].reason instanceof AggregateError);
  assert.equal(results[2].status, "rejected");
  assert.ok(results[2].reason instanceof AggregateError);

  await assert.rejects(isolated.initRife(),
    (error) => error instanceof AggregateError && /invalidation barrier/.test(error.message));
  const disposal = isolated.disposeRife();
  assert.strictEqual(isolated.disposeRife(), disposal);
  await assert.rejects(disposal,
    (error) => error instanceof AggregateError && /session disposal/.test(error.message));
  assert.equal(creates, 1, "failed cleanup must block automatic session recovery");
  assert.equal(releases, 1, "the invalidation claimant releases the session once");
  assert.equal(isolated.isReady(), false);
  assert.equal(isolated.getOrtDevice(), null);
});
