import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const targetDir = resolve(root, "shaders", "upstream");
const sources = [
  {
    name: "FSRCNNX_x2_16-0-4-1.glsl",
    url: "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl",
    sha256: "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
  },
  {
    name: "SSimDownscaler.glsl",
    url: "https://gist.githubusercontent.com/igv/36508af3ffc84410fe39761d6969be10/raw/38992bce7f9ff844f800820df0908692b65bb74a/SSimDownscaler.glsl",
    sha256: "f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804",
  },
  {
    name: "ArtCNN_C4F32.glsl",
    url: "https://raw.githubusercontent.com/Artoriuz/ArtCNN/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32.glsl",
    sha256: "f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3",
  },
  {
    name: "ArtCNN_C4F32_DN.glsl",
    url: "https://raw.githubusercontent.com/Artoriuz/ArtCNN/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DN.glsl",
    sha256: "6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c",
  },
  {
    name: "ArtCNN_C4F32_DS.glsl",
    url: "https://raw.githubusercontent.com/Artoriuz/ArtCNN/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DS.glsl",
    sha256: "a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e",
  },
];

await mkdir(targetDir, { recursive: true });
for (const source of sources) {
  const response = await fetch(source.url, { redirect: "follow" });
  if (!response.ok) throw new Error(`${source.name}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== source.sha256) throw new Error(`${source.name}: checksum mismatch (${digest})`);
  await writeFile(resolve(targetDir, source.name), bytes);
  console.log(`${source.name}: verified`);
}
