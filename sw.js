/* 离线缓存：在家连一次电脑打开页面，之后在球场断网也能用。
 *
 * 缓存的东西分两类：
 *   1. 装 Service Worker 时直接预取的（页面、分析代码、姿态模型）
 *   2. 首次使用时顺带缓存的（Pyodide、numpy、OpenCV、MediaPipe —— 这些来自 CDN，
 *      体积大，只能在真正请求到时才抓得到）
 * 装机时预取第一类，第二类在页面等 SW 就绪之后才开始下载，所以都能被拦到。
 */

const VERSION = 'bball-9ffb946da5';   // 打包时替换成内容版本，改了代码缓存会自动更新


// 预取：页面 + 分析代码 + 姿态模型
const SHELL = [
  './index.html',
  './bball/__init__.py',
  './bball/config.py',
  './bball/pose.py',
  './bball/shots.py',
  './bball/metrics.py',
  './bball/feedback.py',
  './bball/live.py',
  './bball/ball.py',
  './models/pose_landmarker_full.task',
];

// 不缓存：本地测试用的大视频文件
const SKIP = /\.(mp4|mov|avi)$/i;

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    // 逐个加，某个失败不影响其他（比如模型还没下载到电脑上）
    await Promise.all(SHELL.map((u) => c.add(u).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)));
    await self.clients.claim();          // 立刻接管当前页面，后续请求才拦得到
    // 已经在打开的旧页面：它自己不带更新逻辑（那是新版本才有的），
    // 所以这里主动把它导航一次，让它切到新版本。否则用户会一直卡在旧界面上。
    const pages = await self.clients.matchAll({ type: 'window' });
    pages.forEach((c) => { try { c.navigate(c.url); } catch (_) {} });
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || SKIP.test(new URL(req.url).pathname)) return;
  e.respondWith((async () => {
    const hit = await caches.match(req, { ignoreSearch: false });
    if (hit) return hit;
    // 打开页面时地址是 /bball-trainer/，而缓存里的键是 /bball-trainer/index.html，
    // 直接匹配会落空 —— 断网时就打不开了。导航请求单独兜一下首页。
    if (req.mode === 'navigate') {
      const page = await caches.match('./index.html');
      if (page) return page;
    }
    try {
      const res = await fetch(req);
      // 缓存同源资源和 CDN 资源（只缓存正常的完整响应）
      // 只缓存能读到的响应：opaque（跨域脚本标签之类）读不到内容，也验不了
      if (res && res.status === 200 && (res.type === 'basic' || res.type === 'cors')) {
        const copy = res.clone();
        caches.open(VERSION).then((c) => c.put(req, copy)).catch(() => {});
      }
      return res;
    } catch (err) {
      // 断网：导航请求返回应用页面，让它至少能打开
      if (req.mode === 'navigate') {
        const page = await caches.match('./index.html');
        if (page) return page;
      }
      throw err;
    }
  })());
});

// 页面问「缓存够不够离线用」，返回条数和体积
self.addEventListener('message', (e) => {
  if (!e.data || e.data.type !== 'cache-status') return;
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    const keys = await c.keys();
    let bytes = 0;
    for (const req of keys) {
      const res = await c.match(req);
      if (!res) continue;
      const len = res.headers.get('content-length');
      if (len) { bytes += Number(len); continue; }
      try { bytes += (await res.clone().blob()).size; } catch (_) {}
    }
    // 顺便看关键的那几样齐没齐（用户最想知道"还差什么、还差多少"）
    const urls = keys.map((r) => r.url);
    const has = (pat) => urls.some((u) => pat.test(u));
    const key = {
      pyodide: has(/pyodide\.js$/) && has(/pyodide\.asm\.wasm$/),
      numpy: has(/numpy-.*\.whl$/),
      opencv: has(/opencv_python-.*\.whl$/),
      mediapipe: has(/tasks-vision.*vision_bundle\.mjs$/),
      model: has(/pose_landmarker_full\.task$/),
      code: has(/bball\/ball\.py$/),
    };
    e.source && e.source.postMessage({ type: 'cache-status', count: keys.length, bytes, key });
  })());
});
