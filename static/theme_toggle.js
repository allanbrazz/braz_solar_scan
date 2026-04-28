(function () {
  var STORAGE_KEY = 'solar-scan-theme';

  function getTheme() {
    var attr = document.documentElement.getAttribute('data-theme');
    return attr === 'dark' ? 'dark' : 'light';
  }

  function setTheme(theme) {
    var finalTheme = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', finalTheme);
    try {
      localStorage.setItem(STORAGE_KEY, finalTheme);
    } catch (e) {}
    updateToggleState(finalTheme);
  }

  function updateToggleState(theme) {
    var btn = document.querySelector('.theme-toggle-btn');
    if (!btn) return;

    var isDark = theme === 'dark';
    btn.setAttribute('aria-pressed', isDark ? 'true' : 'false');

    var label = btn.querySelector('.theme-toggle-label');
    if (label) {
      label.textContent = isDark ? 'Modo claro' : 'Modo escuro';
    }

    btn.setAttribute(
      'aria-label',
      isDark ? 'Alternar para modo claro' : 'Alternar para modo escuro'
    );
  }

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.querySelector('.theme-toggle-btn');
    updateToggleState(getTheme());

    if (btn) {
      btn.addEventListener('click', function () {
        var next = getTheme() === 'dark' ? 'light' : 'dark';
        setTheme(next);
      });
    }
  });
})();
