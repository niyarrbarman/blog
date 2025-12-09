// Theme toggle functionality
(function () {
  const STORAGE_KEY = "theme";
  const LIGHT_VALUE = "light";
  const DARK_VALUE = "dark";
  const DATA_ATTRIBUTE = "data-theme";

  // Get stored theme or default to light
  function getStoredTheme() {
    return localStorage.getItem(STORAGE_KEY) || LIGHT_VALUE;
  }

  // Set theme
  function setTheme(theme) {
    document.documentElement.setAttribute(DATA_ATTRIBUTE, theme);
    localStorage.setItem(STORAGE_KEY, theme);
    updateThemeIcon(theme);
    updateHighlightTheme(theme);
  }

  // Update theme toggle icon
  function updateThemeIcon(theme) {
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      const icon = toggle.querySelector(".theme-icon");
      if (icon) {
        icon.textContent = theme === DARK_VALUE ? "☀" : "☾";
      }
    }
  }

  // Update highlight.js theme
  function updateHighlightTheme(theme) {
    const lightTheme = document.getElementById("hljs-light");
    const darkTheme = document.getElementById("hljs-dark");

    if (lightTheme && darkTheme) {
      if (theme === DARK_VALUE) {
        lightTheme.disabled = true;
        darkTheme.disabled = false;
      } else {
        lightTheme.disabled = false;
        darkTheme.disabled = true;
      }
    }
  }

  // Toggle theme
  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute(DATA_ATTRIBUTE);
    const newTheme = currentTheme === DARK_VALUE ? LIGHT_VALUE : DARK_VALUE;
    setTheme(newTheme);
  }

  // Initialize theme on page load
  function initTheme() {
    const storedTheme = getStoredTheme();
    setTheme(storedTheme);

    // Add click listener to toggle button
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      toggle.addEventListener("click", toggleTheme);
    }
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTheme);
  } else {
    initTheme();
  }
})();

// Highlight.js initialization
(function () {
  function initHighlight() {
    if (typeof hljs !== "undefined") {
      hljs.highlightAll();
      addCopyButtons();
    }
  }

  // Add copy buttons to code blocks
  function addCopyButtons() {
    const codeBlocks = document.querySelectorAll("pre code");

    codeBlocks.forEach(function (codeBlock) {
      const pre = codeBlock.parentElement;

      // Create copy button
      const copyButton = document.createElement("button");
      copyButton.className = "copy-button";
      copyButton.textContent = "⎘";
      copyButton.title = "Copy code";
      copyButton.setAttribute("aria-label", "Copy code to clipboard");

      // Add click handler
      copyButton.addEventListener("click", function () {
        const code = codeBlock.textContent;

        navigator.clipboard
          .writeText(code)
          .then(function () {
            copyButton.textContent = "✓";
            copyButton.classList.add("copied");

            setTimeout(function () {
              copyButton.textContent = "⎘";
              copyButton.classList.remove("copied");
            }, 2000);
          })
          .catch(function (err) {
            console.error("Failed to copy:", err);
            copyButton.textContent = "✗";

            setTimeout(function () {
              copyButton.textContent = "⎘";
            }, 2000);
          });
      });

      // Insert button into pre element
      pre.style.position = "relative";
      pre.insertBefore(copyButton, codeBlock);
    });
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initHighlight);
  } else {
    initHighlight();
  }
})();

// KaTeX auto-render initialization
(function () {
  function initKaTeX() {
    if (typeof renderMathInElement !== "undefined") {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    }
  }

  // Wait for KaTeX to load
  function waitForKaTeX() {
    if (typeof renderMathInElement !== "undefined") {
      initKaTeX();
    } else {
      // Check again after a short delay
      setTimeout(waitForKaTeX, 100);
    }
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", waitForKaTeX);
  } else {
    waitForKaTeX();
  }
})();

// Smooth scroll for anchor links
(function () {
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
      anchor.addEventListener("click", function (e) {
        const href = this.getAttribute("href");

        if (href === "#") return;

        const target = document.querySelector(href);

        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });

          // Update URL hash without scrolling
          history.pushState(null, null, href);
        }
      });
    });
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSmoothScroll);
  } else {
    initSmoothScroll();
  }
})();
