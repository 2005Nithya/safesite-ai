const CACHE_NAME = 'safesite-ai-v1';
const APP_SHELL = ['/', '/auth', '/dashboard', '/detection', '/static/logo.jpeg', '/static/manifest.json'];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL)));
});

self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key)))));
});

self.addEventListener('fetch', event => {
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request).then(response => response || caches.match('/'))));
});
