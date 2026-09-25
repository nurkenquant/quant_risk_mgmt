(function () {
  "use strict";
  var MAX_SIDE = 2000;

  // Shrink phone photos before upload: faster on mobile data and within API limits.
  function shrink(file) {
    if (!file || !/^image\/(jpeg|png|webp|heic|heif)$/i.test(file.type) || file.size < 900 * 1024) {
      return Promise.resolve(file);
    }
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () {
        var scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
        var c = document.createElement("canvas");
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(url);
        c.toBlob(function (blob) {
          if (!blob) return resolve(file);
          resolve(new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" }));
        }, "image/jpeg", 0.85);
      };
      img.onerror = function () { URL.revokeObjectURL(url); resolve(file); };
      img.src = url;
    });
  }

  function replaceFile(input, file) {
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch (e) { /* old browsers: upload the original */ }
  }

  function showBusy() { var b = document.getElementById("busy"); if (b) b.hidden = false; }

  document.querySelectorAll('input[type=file]').forEach(function (input) {
    input.addEventListener("change", function () {
      var f = input.files && input.files[0];
      if (!f) return;
      shrink(f).then(function (small) {
        if (small !== f) replaceFile(input, small);
        if (input.classList.contains("auto-submit")) {
          showBusy();
          input.form.submit();
        } else {
          var drop = input.closest(".drop");
          if (drop) {
            drop.classList.add("has-file");
            drop.querySelector("b").textContent = f.name;
          }
        }
      });
    });
  });

  // Uploading a receipt on the add/edit form also takes a moment.
  document.querySelectorAll("form[enctype='multipart/form-data']").forEach(function (form) {
    form.addEventListener("submit", function () {
      var fi = form.querySelector("input[type=file]");
      if (fi && fi.files && fi.files.length) showBusy();
    });
  });

  // Bottom sheet
  var fab = document.getElementById("fab"), sheet = document.getElementById("sheet");
  if (fab && sheet) {
    var toggle = function (open) {
      sheet.hidden = !open;
      fab.setAttribute("aria-expanded", open ? "true" : "false");
    };
    fab.addEventListener("click", function () { toggle(sheet.hidden); });
    sheet.addEventListener("click", function (e) { if (e.target === sheet) toggle(false); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") toggle(false); });
  }

  // Swap category list when switching expense/revenue
  var sel = document.querySelector("select[data-expense]");
  document.querySelectorAll(".kind-toggle input").forEach(function (r) {
    r.addEventListener("change", function () {
      if (!sel) return;
      var list = JSON.parse(sel.dataset[r.value]);
      var cur = sel.value;
      sel.innerHTML = "";
      list.forEach(function (c) {
        var o = document.createElement("option");
        o.textContent = c;
        if (c === cur) o.selected = true;
        sel.appendChild(o);
      });
      var h = document.querySelector(".top h1");
      if (h && /^Add/.test(h.textContent)) h.textContent = r.value === "income" ? "Add revenue" : "Add expense";
    });
  });

  // Chart hover / tap readout
  document.querySelectorAll(".bars .col").forEach(function (col) {
    var show = function () {
      var card = col.closest(".card"), tip = card && card.querySelector(".tip");
      if (!tip) return;
      card.querySelectorAll(".col.sel").forEach(function (c) { c.classList.remove("sel"); });
      col.classList.add("sel");
      tip.textContent = col.dataset.tip;
    };
    col.addEventListener("mouseenter", show);
    col.addEventListener("focus", show);
    col.addEventListener("click", show);
  });

  // Confirm destructive actions
  document.querySelectorAll("form[data-confirm]").forEach(function (f) {
    f.addEventListener("submit", function (e) { if (!confirm(f.dataset.confirm)) e.preventDefault(); });
  });

  // Live monthly-payment preview on the loan form
  var lf = document.getElementById("loan-form"), prev = document.getElementById("loan-preview");
  if (lf && prev) {
    var num = function (n) { return parseFloat(String(lf.elements[n].value).replace(/\s/g, "").replace(",", ".")); };
    var calc = function () {
      var P = num("principal"), rate = num("rate"), n = parseInt(lf.elements.term_months.value, 10);
      if (!(P > 0) || !(rate >= 0) || !(n > 0)) return;
      var r = rate / 1200;
      var pmt = num("payment") || (r ? P * r / (1 - Math.pow(1 + r, -n)) : P / n);
      var extra = num("extra") || 0, bal = P, interest = 0, k = 0;
      while (bal > 0.005 && k < n * 3 + 600) {
        var i = bal * r, pr = Math.min(pmt - i, bal);
        if (pr <= 0 && extra <= 0) { prev.textContent = "Payment is below the monthly interest — the loan would never be repaid."; return; }
        interest += i; bal -= pr + Math.min(extra, Math.max(bal - pr, 0)); k++;
      }
      var fmt = function (v) { return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }); };
      prev.textContent = "≈ " + fmt(pmt + extra) + " / month · " + k + " payments · total interest " + fmt(interest);
    };
    lf.addEventListener("input", calc);
    calc();
  }
})();
