import test from "node:test";
import assert from "node:assert/strict";

import * as rife from "../fsrcnnx-rife.js";

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

test("RIFE model switching commits the latest replacement before releasing the old session", async (t) => {
  const events = [];
  const requests = [];
  const recoveredLoss = deferred();
  const devices = {
    initial: { id: "device-initial" },
    stale: { id: "device-stale" },
    latest: { id: "device-latest" },
    recovered: { id: "device-recovered", lost: recoveredLoss.promise },
  };
  const releases = { initial: 0, stale: 0, latest: 0, same: 0, recovered: 0 };

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
  let createCount = 0;
  const state = {
    env: { wasm: {}, webgpu: { enableFp16: false, device: null } },
    create(url) {
      createCount++;
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

  assert.equal(rife.setModel("rife_orig"), true);
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
  assert.equal(await rife.confirmOrtDeviceAdopted(devices.latest), true);
  assert.equal(releases.initial, 1);
  const initialReleaseIndex = events.findIndex(({ type }) => type === "release-initial");
  assert.ok(latestCreateIndex >= 0 && latestCreateIndex < initialReleaseIndex);
  const initialRelease = events[initialReleaseIndex];
  assert.equal(initialRelease.activeDevice, devices.latest);
  assert.equal(initialRelease.ready, true);

  // A subsequent replacement on the already-adopted device can release its
  // predecessor immediately because the candidate itself preserves that device.
  assert.equal(rife.setModel("rife_orig"), true);
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
});
