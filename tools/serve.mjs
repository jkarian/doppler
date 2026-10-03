// Zero-dependency dev server. Serves the project folder and strips TypeScript
// types on the fly with Node's built-in stripTypeScriptTypes (Node 23.2+).
//
//   node tools/serve.mjs [port]

import { createServer } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";
import { basename, extname, join, normalize, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const port = Number(process.argv[2] ?? 5173);

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".ts": "text/javascript; charset=utf-8",
  ".wgsl": "text/plain; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".py": "text/plain; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".bin": "application/octet-stream",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".mp4": "video/mp4",
  ".ogg": "audio/ogg",
  ".flac": "audio/flac",
};

createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  if (req.method === "POST" && url.pathname === "/capture") return saveCapture(req, res, url);
  if (req.method === "POST" && url.pathname === "/graph") return saveGraph(req, res, url);
  let path = normalize(decodeURIComponent(url.pathname));
  if (path.endsWith("/") || path.endsWith("\\")) path = join(path, "index.html");
  const file = join(root, path);
  if (!file.startsWith(root)) return void res.writeHead(403).end();
  try {
    let body = await readFile(file);
    const ext = extname(file).toLowerCase();
    if (ext === ".ts") body = stripTypeScriptTypes(body.toString("utf8"), { mode: "strip" });
    const headers = { "content-type": types[ext] ?? "application/octet-stream", "cache-control": "no-store", "accept-ranges": "bytes" };
    // Byte ranges: browsers need them to seek in audio and video.
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
    if (range) {
      const size = body.length;
      let start = range[1] ? Number(range[1]) : size - Number(range[2]);
      let end = range[1] && range[2] ? Number(range[2]) : size - 1;
      start = Math.max(0, start);
      end = Math.min(end, size - 1);
      if (start > end) return void res.writeHead(416, { "content-range": `bytes */${size}` }).end();
      res.writeHead(206, { ...headers, "content-range": `bytes ${start}-${end}/${size}`, "content-length": end - start + 1 });
      return void res.end(body.subarray(start, end + 1));
    }
    res.writeHead(200, headers);
    res.end(body);
  } catch (err) {
    res.writeHead(err.code === "ENOENT" ? 404 : 500, { "content-type": "text/plain" });
    res.end(String(err.message ?? err));
  }
}).listen(port, () => console.log(`doppler dev server: http://localhost:${port}/`));

// Display page frame captures land in captures/ (clip frames in captures/<dir>/).
async function saveCapture(req, res, url) {
  const name = basename(url.searchParams.get("name") ?? `capture-${Date.now()}.png`);
  const dir = join(root, "captures", basename(url.searchParams.get("dir") ?? ""));
  if (!/\.(png|jpg|json)$/.test(name)) return void res.writeHead(400).end();
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, name), Buffer.concat(chunks));
  if (name.endsWith(".png")) console.log(`saved ${join(dir, name)}`);
  res.writeHead(200).end();
}

// The graph editor saves graphs/<name>.json here.
async function saveGraph(req, res, url) {
  const name = url.searchParams.get("name") ?? "";
  if (!/^[\w-]{1,64}$/.test(name)) return void res.writeHead(400).end("bad graph name");
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString("utf8");
  try {
    const graph = JSON.parse(text);
    if (graph.version !== 1 || !Array.isArray(graph.nodes) || !Array.isArray(graph.wires)) throw new Error("not a graph");
  } catch (err) {
    return void res.writeHead(400).end(String(err.message ?? err));
  }
  await mkdir(join(root, "graphs"), { recursive: true });
  await writeFile(join(root, "graphs", `${name}.json`), text);
  console.log(`saved graphs/${name}.json`);
  res.writeHead(200).end();
}
