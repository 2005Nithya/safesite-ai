/* SafeSite AI - shared app JS */
(function () {
  // Apply saved theme before paint
  var savedTheme = localStorage.getItem('safesite-theme');
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  // Theme toggler
  var themeToggles = document.querySelectorAll('[data-theme-toggle]');
  themeToggles.forEach(function (toggle) {
    if (document.documentElement.getAttribute('data-theme') === 'dark') {
      toggle.checked = true;
    }
    toggle.addEventListener('change', function () {
      var dark = toggle.checked;
      document.documentElement.setAttribute('data-theme', dark ? 'dark' : '');
      localStorage.setItem('safesite-theme', dark ? 'dark' : 'light');
      if (window.showToast) {
        window.showToast(dark ? 'Dark theme enabled' : 'Light theme enabled');
      }
    });
  });

  // Notification preferences
  var alertToggle = document.querySelectorAll('[data-alert-toggle]');
  alertToggle.forEach(function (toggle) {
    if (localStorage.getItem('safesite-alerts') === 'on') {
      toggle.checked = true;
    }
    toggle.addEventListener('change', function () {
      localStorage.setItem('safesite-alerts', toggle.checked ? 'on' : 'off');
    });
  });

  var emailToggle = document.querySelectorAll('[data-email-toggle]');
  emailToggle.forEach(function (toggle) {
    if (localStorage.getItem('safesite-email-alerts') === 'on') {
      toggle.checked = true;
    }
    toggle.addEventListener('change', function () {
      localStorage.setItem('safesite-email-alerts', toggle.checked ? 'on' : 'off');
    });
  });

  // Animated counters
  var counters = document.querySelectorAll('[data-count]');
  counters.forEach(function (el) {
    var target = parseFloat(el.getAttribute('data-count')) || 0;
    var suffix = el.getAttribute('data-suffix') || '';
    var duration = 900;
    var start = null;
    function step(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });

  // Toast helper
  window.showToast = function (message, type) {
    var existing = document.querySelector('.toast');
    if (existing) existing.remove();
    var toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    if (type === 'error') toast.style.borderLeftColor = '#ef4444';
    document.body.appendChild(toast);
    setTimeout(function () { toast.classList.add('show'); }, 10);
    setTimeout(function () {
      toast.classList.remove('show');
      setTimeout(function () { toast.remove(); }, 300);
    }, 2600);
  };
})();