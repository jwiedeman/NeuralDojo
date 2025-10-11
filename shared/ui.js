(() => {
  function $(selector, root = document) {
    return root.querySelector(selector);
  }

  function $all(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function defaultFormat(value) {
    if (typeof value === 'number') return value.toString();
    return value;
  }

  function bindRangeControl(input, {
    valueEl,
    format = defaultFormat,
    onInput,
    onCommit
  } = {}) {
    if (!input) return () => {};
    const emit = (cb, v) => {
      if (typeof cb === 'function') cb(v, input);
    };
    const update = () => {
      const raw = input.value;
      if (valueEl) {
        valueEl.textContent = format(parseFloat(raw), raw);
      }
      emit(onInput, parseFloat(raw));
    };
    const commit = () => {
      const raw = input.value;
      if (valueEl) {
        valueEl.textContent = format(parseFloat(raw), raw);
      }
      emit(onCommit, parseFloat(raw));
    };
    input.addEventListener('input', update);
    input.addEventListener('change', commit);
    update();
    return { update, commit };
  }

  function initTabs(root = document) {
    const tabButtons = $all('.tab-button', root);
    const tabPanels = $all('.tab-panel', root);
    if (!tabButtons.length) return () => {};

    const activate = name => {
      if (!name) return;
      tabButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === name));
      tabPanels.forEach(panel => panel.classList.toggle('active', panel.dataset.tab === name));
    };

    const initial = tabButtons.find(btn => btn.classList.contains('active'))?.dataset.tab || tabButtons[0].dataset.tab;
    activate(initial);
    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => activate(btn.dataset.tab));
    });
    return activate;
  }

  function toggleSections(sections, activeName) {
    sections.forEach(({ name, el }) => {
      if (!el) return;
      el.classList.toggle('active', name === activeName);
    });
  }

  window.NeuralUI = {
    bindRangeControl,
    initTabs,
    toggleSections,
    $, $all
  };
})();
