/* The capsule. Python owns the state machine and calls show()/hide(); this page
   draws the state, animates every shape with springs, and reports each frame's
   rectangles back so the native glass under the page and the click-through
   regions of the panel follow exactly. */
(() => {
'use strict';
const { I, esc, $, post, on, env, L, reduceMotion, Value } = kit;

const stage = $('#stage'), capEl = $('#cap'), budsEl = $('#buds'), handleEl = $('#handle'), hbEl = $('#hb');
const MARGIN = 14;          // gap between the capsule and the bottom of the visible frame
const GAP = 8;
const PILL_H = 40, BUD_H = 36;
const QUIET_MS = 3000;      // "no sound detected" after this long with no voice yet
const VOICE = 0.08;         // analyzer level that counts as voice
const NB = 13;
const PROF = [1, .92, .8, .66, .52, .4, .3, .22];

const S = {
  st: 'hidden', d: {}, kind: 'pill', hidden: true, layer: null,
  mode: 'click', recAt: 0, max: 900, cd: false, quiet: false, heard: false, lastVoice: 0,
  level: 0, levelAt: 0, tmrText: '', say: '',
  anchor: { mode: 'bottom', x: null, y: null }, last: null,
  handleOn: false, handleHover: false, hover: false, hovBtn: null, native: false, lastGeo: '',
};
let seq = 0;

/* ---------------- shapes ---------------- */
class Shape {
  constructor(id, el, glass = true) {
    this.id = id; this.el = el; this.glass = glass; this.gone = false;
    this.x = new Value(0); this.y = new Value(0); this.w = new Value(40); this.h = new Value(40); this.r = new Value(20);
    this.o = new Value(0); this.s = new Value(1); this.tx = new Value(0); this.ty = new Value(0);
  }
  vals() { return [this.x, this.y, this.w, this.h, this.r, this.o, this.s, this.tx, this.ty]; }
  busy() { return this.vals().some(v => v.busy()); }
  place(r) { this.x.set(r.x); this.y.set(r.y); this.w.set(r.w); this.h.set(r.h); if (r.r != null) this.r.set(r.r); }
  go(r, spring, now, delay = 0) {
    this.x.animate(r.x, spring, now, delay); this.y.animate(r.y, spring, now, delay);
    this.w.animate(r.w, spring, now, delay); this.h.animate(r.h, spring, now, delay);
    if (r.r != null) this.r.animate(r.r, spring === 'morph' ? 'card' : spring, now, delay);
  }
  apply(now) {
    const f = {
      x: this.x.value(now), y: this.y.value(now), w: Math.max(0, this.w.value(now)), h: Math.max(0, this.h.value(now)),
      r: Math.max(0, this.r.value(now)), o: Math.min(1, Math.max(0, this.o.value(now))), s: Math.max(0, this.s.value(now)),
      tx: this.tx.value(now), ty: this.ty.value(now),
    };
    const st = this.el.style;
    st.width = f.w.toFixed(2) + 'px'; st.height = f.h.toFixed(2) + 'px';
    st.borderRadius = f.r.toFixed(2) + 'px'; st.opacity = f.o.toFixed(3);
    st.transform = `translate(${(f.x + f.tx).toFixed(2)}px, ${(f.y + f.ty).toFixed(2)}px) scale(${f.s.toFixed(4)})`;
    return f;
  }
  geo(f) {
    const w = f.w * f.s, h = f.h * f.s;
    const cx = f.x + f.tx + f.w / 2, cy = f.y + f.ty + f.h / 2;
    return { id: this.id, x: +(cx - w / 2).toFixed(2), y: +(cy - h / 2).toFixed(2), w: +w.toFixed(2), h: +h.toFixed(2), r: +(f.r * f.s).toFixed(2), a: +f.o.toFixed(3), glass: this.glass, hit: !this.gone && f.o > .05 };
  }
}
const cap = new Shape('cap', capEl);
const hb = new Shape('hb', hbEl);
hb.place({ x: 0, y: 0, w: 40, h: 40, r: 20 });
let buds = [];

/* ---------------- views ---------------- */
const tail = s => { const t = Array.from(String(s || '')); return t.length > 24 ? '…' + t.slice(-24).join('') : t.join(''); };
const fmt = (sec, up) => { const s = Math.max(0, up ? Math.ceil(sec - 1e-4) : Math.floor(sec)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const waveHTML = idle => `<span class="wave${idle ? ' idle' : ''}">${'<i></i>'.repeat(NB)}</span>`;
const CHECK = `<span class="lead ok"><svg class="i draw" viewBox="0 0 24 24" aria-hidden="true" style="stroke-width:2.4"><path d="M5 12.5l4.3 4.3L19 7.2"/></svg></span>`;
// Built when drawn, in the language of the moment.
const BUD = {
  undo: () => `<button data-act="undo">${I('undo')}<span>${L('撤销', 'Undo')}</span></button>`,
  edit: () => `<button data-act="edit">${I('pencil')}<span>${L('修改', 'Edit')}</span></button>`,
  rerefine: () => `<button class="pri" data-act="rerefine">${I('refresh')}<span>${L('重新润色', 'Refine Again')}</span></button>`,
  restart: () => `<button class="pri" data-act="restart">${I('refresh')}<span>${L('重新打开', 'Relaunch')}</span></button>`,
  billing: () => `<button class="pri" data-act="billing">${I('key')}<span>${L('去充值…', 'Add Credit…')}</span></button>`,
  setkey: () => `<button class="pri" data-act="setkey">${I('key')}<span>${L('设置 API Key…', 'Set API Key…')}</span></button>`,
  input: () => `<button data-act="input">${I('mic')}<span>${L('选择输入…', 'Choose Input…')}</span></button>`,
  micperm: () => `<button class="pri" data-act="micperm">${I('mic')}<span>${L('打开设置…', 'Open Settings…')}</span></button>`,
  perm: () => `<button class="pri" data-act="perm">${I('a11y')}<span>${L('打开设置…', 'Open Settings…')}</span></button>`,
  log: () => `<button data-act="log">${I('doc')}<span>${L('显示日志', 'Show Log')}</span></button>`,
};
const extendBud = min => `<button data-act="extend">${I('plus')}<span>${L(`延长 ${min | 0} 分钟`, `${min | 0} More Minutes`)}</span></button>`;
const why = k => ({
  timeout: L('润色超时', 'refinement timed out'), error: L('润色失败', 'refinement failed'),
  truncated: L('润色被截断', 'refinement was cut off'), empty: L('润色没有返回', 'refinement came back empty'),
  credit: L('OpenAI 账户余额用完了', 'your OpenAI account is out of credit'), badkey: L('API Key 无效', 'the API key was rejected'),
})[k] || L('润色失败', 'refinement failed');
const etaText = eta => eta ? ` · ${esc(eta)}` : '';
// Announced in steps of ten, not on every progress report.
const downloadSay = p => L(`语音模型下载中，${Math.floor(p * 10) * 10}%`, `Downloading the speech model, ${Math.floor(p * 10) * 10}%`);
const btnCancel = () => `<button class="ib" data-act="cancel" aria-label="${L('取消', 'Cancel')}">${I('xmark')}</button>`;
const btnFinish = () => `<button class="ib pri" data-act="finish" aria-label="${L('完成', 'Done')}">${I('check')}</button>`;
const units = n => L(`${n | 0} 字`, `${n | 0} word${(n | 0) === 1 ? '' : 's'}`);

function cardHTML(title, sub, text, btns) {
  return `<div class="card-in"><div class="card-h">${I('doc')}<span class="t">${esc(title)}<small>${esc(sub)}</small></span><button class="ib sm" data-act="close" aria-label="${L('关闭', 'Close')}">${I('xmark')}</button></div>`
    + `<textarea class="card-field" id="cardField" rows="3" spellcheck="false" aria-label="${L('听写文字', 'Dictated text')}">${esc(text || '')}</textarea>`
    + `<div class="card-f"><span class="kh">${btns.length > 1 ? L('⏎ 替换 · ⇧⏎ 换行 · esc 取消', '⏎ replace · ⇧⏎ new line') : L('⏎ 完成 · ⇧⏎ 换行 · esc 取消', '⏎ done · ⇧⏎ new line · esc close')}</span><span class="sp"></span>`
    + btns.map(([a, l, p]) => `<button class="pbtn${p ? ' pri' : ''}" data-act="${a}">${esc(l)}</button>`).join('') + `</div></div>`;
}

// English practice: the sentence, then the phrases worth keeping.
// Long English scrolls in place from the top; a fade at the bottom says there is more.
function enFade(t) { t.classList.toggle('more', t.scrollTop + t.clientHeight < t.scrollHeight - 1); }
document.addEventListener('scroll', e => { if (e.target.classList?.contains('en-t')) enFade(e.target); }, true);
// Each English word and each phrase row is a button for the word book (data-w = the term).
// Words carry their offset in the sentence (data-i); everything between words is a .gp span, so a dragged block can be highlighted whole.
function words(t) {
  let out = '', at = 0;
  for (const m of t.matchAll(/[A-Za-z][A-Za-z'’-]*/g)) {
    if (m.index > at) out += `<span class="gp">${esc(t.slice(at, m.index))}</span>`;
    out += `<span class="wd" data-w="${m[0]}" data-k="word" data-i="${m.index}">${m[0]}</span>`;
    at = m.index + m[0].length;
  }
  return out + (at < t.length ? `<span class="gp">${esc(t.slice(at))}</span>` : '');
}
function englishHTML(o) {
  const rows = (o.phrases || []).map(p => `<div class="en-row wd" data-w="${esc(p.en)}" data-k="phrase"><b>${esc(p.en)}</b><span>${esc(p.zh)}</span></div>`).join('');
  return `<div class="card-in en-card"><div class="en-h"><span class="en-tag">EN</span><span class="sp"></span>`
    + `<button class="ib sm" data-act="copyen" aria-label="${L('复制英文', 'Copy English')}">${I('copy')}</button>`
    + `<button class="ib sm" data-act="close" aria-label="${L('关闭', 'Close')}">${I('xmark')}</button></div>`
    + `<div class="en-t">${words(o.en)}</div>` + (rows ? `<div class="en-p">${rows}</div>` : '')
    + (o.use ? `<div class="card-f"><span class="sp"></span><button class="pbtn pri" data-act="useen">${L('用这句替换', 'Use this')}</button></div>` : '')
    + `</div>`;
}

function recHTML() {
  const o = S.d;
  const sel = o.sel ? `<span class="chip">${I('cursor')}${L('改写所选', 'Rewrite selection')}</span>` : '';
  const duck = o.ducked ? `<span class="lockg" aria-label="${L('已静音其他声音', 'Other sound muted')}">${I('speaker-slash')}</span>` : '';
  const lock = `<span class="lockg">${I('lock')}</span>`;
  const t = (performance.now() - S.recAt) / 1000;
  if (S.cd) {
    const rem = S.max - t, p = Math.max(0, Math.min(1, rem / 60));
    S.tmrText = fmt(rem, true);
    return [`${btnCancel()}${waveHTML()}<span class="tmr warn">${S.tmrText}</span><span class="ringwrap"><svg class="ring" viewBox="0 0 36 36" aria-hidden="true"><circle class="bg" cx="18" cy="18" r="16.5"/><circle class="fg" cx="18" cy="18" r="16.5" stroke-dasharray="103.7" stroke-dashoffset="${(103.7 * (1 - p)).toFixed(1)}"/></svg>${btnFinish()}</span>`,
      L(`还剩 ${Math.ceil(rem)} 秒，到点会自动结束${o.ext > 0 ? `，可以延长 ${o.ext | 0} 分钟` : ''}`,
        `${Math.ceil(rem)} seconds left; it will finish by itself${o.ext > 0 ? `, or add ${o.ext | 0} minutes` : ''}`)];
  }
  S.tmrText = fmt(t);
  if (S.quiet && S.mode !== 'hold') {
    return [`${btnCancel()}${S.mode === 'latch' ? lock : ''}${waveHTML(true)}<span class="hint warn">${L('没有检测到声音', 'No sound detected')}</span>${btnFinish()}`,
      L('没有检测到声音，检查一下麦克风', 'No sound detected. Check the microphone.')];
  }
  const selSay = o.sel ? L('，将改写所选文字', ', rewriting the selection') : '';
  switch (S.mode) {
    case 'hold': return [`<span class="recdot"></span>${sel}${waveHTML()}<span class="hint">${L('松开完成', 'Release to finish')}</span>`, L('录音中，松开按键完成', 'Recording. Release the key to finish') + selSay];
    case 'latch': return [`${btnCancel()}${lock}${sel}${duck}${waveHTML()}<span class="tmr">${S.tmrText}</span>${btnFinish()}`, L('已锁定录音，再点一下右 ⌘ 完成', 'Recording locked. Tap right ⌘ again to finish') + selSay];
    default: return [`${btnCancel()}${sel}${duck}${waveHTML()}<span class="tmr">${S.tmrText}</span>${btnFinish()}`, L('录音中，再点一下右 ⌘ 完成', 'Recording. Tap right ⌘ again to finish') + selSay];
  }
}

// [html, kind, buds, what VoiceOver says]
function view(st, o) {
  switch (st) {
    case 'starting': return [`<span class="solo">${I('mic', 'breathe')}</span>`, 'circle', [], L('正在打开麦克风', 'Opening the microphone')];
    // The last minute buds a 延长 button off the capsule.
    case 'rec': { const [h, say] = recHTML(); return [h, 'pill', S.cd && o.ext > 0 ? [extendBud(o.ext)] : [], say]; }
    case 'transcribing': return [`<span class="dots"><i></i><i></i><i></i></span><span class="lbl">${L('转写中', 'Transcribing')}</span>`, 'pill', [], L('转写中', 'Transcribing')];
    case 'refining': return [`<span class="spark">${I('sparkles')}</span><span class="raw shimmer">${esc(tail(o.raw))}</span>`, 'pill', [], L('润色中', 'Refining')];
    case 'inserted': {
      const done = o.replaced ? L('已替换所选', 'Replaced selection') : L('已插入', 'Inserted');
      return [`${CHECK}<span class="lbl">${done} · ${units(o.n)}</span>`, 'pill', [BUD.undo(), BUD.edit()], `${done} ${units(o.n)}`];
    }
    case 'inserted-raw-net': return [`<span class="lead warn wiggle">${I('warn')}</span><span class="lbl">${L('已插入原始转写', 'Inserted the raw transcript')} <span class="sub">· ${why(o.why)}</span></span>`, 'pill', o.why === 'credit' ? [BUD.billing()] : o.why === 'badkey' ? [BUD.setkey()] : [BUD.rerefine(), BUD.undo()], `${L('已插入原始转写，', 'Inserted the raw transcript: ')}${why(o.why)}`];
    case 'inserted-raw-trial': {
      const r = { trial_region: L('免费试用仅限美国和加拿大', 'free trial is US and Canada only'), trial_paused: L('免费试用本月已暂停', 'free trial paused this month'), trial_ip: L('这个网络今天的免费额度用完了', "this network's free trial is done for today") }[o.why] || L('免费试用已用完', 'free trial used up');
      return [`<span class="lead warn">${I('key')}</span><span class="lbl">${L('已插入原始转写', 'Inserted the raw transcript')} <span class="sub">· ${r}</span></span>`, 'pill', [BUD.setkey()], L(`已插入原始转写，${r}，可以填自己的 API Key`, `Inserted the raw transcript: ${r}; add your own API key`)];
    }
    case 'inserted-raw-key': return [`<span class="lead warn">${I('key')}</span><span class="lbl">${L('已插入原始转写', 'Inserted the raw transcript')} <span class="sub">· ${L('缺少 API Key', 'no API key')}</span></span>`, 'pill', [BUD.setkey()], L('已插入原始转写，还没有设置 API Key', 'Inserted the raw transcript: no API key is set')];
    case 'inserted-unsure': return [`<span class="lead warn">${I('warn')}</span><span class="lbl">${L('可能没插进去', 'May not have gone in')} <span class="sub">· ${L('已复制，⌘V 粘贴', 'copied, press ⌘V')}</span></span>`, 'pill', [], L('可能没插进去，文字已复制，按 ⌘V 粘贴', 'The text may not have gone in. It is copied; press Command V to paste')];
    case 'edit-notarget': return [cardHTML(L('没有可插入的位置', 'Nowhere to insert'), L('已复制到剪贴板。改完按 ⏎ 再复制一次。', 'Copied to the clipboard. Press ⏎ after editing to copy again.'), o.text, [['done', L('完成', 'Done'), true]]), 'card', [], L('没有可插入的位置，结果已复制到剪贴板，可以直接修改', 'Nowhere to insert. The text is copied to the clipboard and can be edited here')];
    case 'edit-modify': return [cardHTML(L('修改刚插入的文字', 'Edit the inserted text'), L('替换 = 在原 App 里撤销那次粘贴，再粘贴新文字', 'Replace undoes the paste in its app, then pastes the new text'), o.text, [['close', L('取消', 'Cancel')], ['replace', L('替换', 'Replace'), true]]), 'card', [], L('修改刚插入的文字，回车替换，esc 取消', 'Edit the inserted text. Return replaces it, Escape cancels')];
    case 'english': return [englishHTML(o), 'card', [], L('英语练习：', 'English practice: ') + (o.en || '')];
    case 'empty': return [`<span class="lead">${I('mic-slash')}</span><span class="lbl">${L('没有听到声音', 'Heard nothing')}${o.device ? ` <span class="sub">· ${esc(o.device)}</span>` : ''}</span>`, 'pill', [], L('没有听到声音', 'Heard nothing')];
    case 'mic': return o.why === 'stuck'
      ? [`<span class="lead err wiggle">${I('warn')}</span><span class="lbl">${L('麦克风卡住了', 'The microphone is stuck')} <span class="sub">· ${L('重新打开就好', 'relaunching fixes it')}</span></span>`, 'pill', [BUD.restart()], L('麦克风卡住了，重新打开就好', 'The microphone is stuck; relaunching the app fixes it')]
      : o.why === 'denied'
      ? [`<span class="lead err wiggle">${I('mic-slash')}</span><span class="lbl">${L('没有麦克风权限', 'No microphone permission')} <span class="sub">· ${L('在系统设置里打开', 'turn it on in System Settings')}</span></span>`, 'pill', [BUD.micperm()], L('没有麦克风权限', 'No microphone permission')]
      : [`<span class="lead err wiggle">${I('warn')}</span><span class="lbl">${L('麦克风不可用', 'Microphone unavailable')} <span class="sub">· ${L('可能被其他 App 占用', 'another app may be using it')}</span></span>`, 'pill', [BUD.input()], L('麦克风不可用', 'Microphone unavailable')];
    case 'download': {
      const p = Math.max(0, Math.min(1, +o.p || 0));
      return [`<span class="pring" style="--p:${p.toFixed(3)}"></span><span class="lbl">${L('语音模型下载中', 'Downloading the speech model')} · <span class="pct">${Math.round(p * 100)}%</span><span class="sub eta">${etaText(o.eta)}</span></span>`, 'pill', [], downloadSay(p)];
    }
    case 'cancelled': return [`<span class="lead">${I('xmark')}</span><span class="lbl">${L('已取消', 'Cancelled')}</span>`, 'pill', [], L('已取消', 'Cancelled')];
    case 'undone': return [`<span class="lead">${I('undo')}</span><span class="lbl">${L('已撤销', 'Undone')}</span>`, 'pill', [], L('已撤销', 'Undone')];
    case 'replaced': return [`${CHECK}<span class="lbl">${L('已替换', 'Replaced')} · ${units(o.n)}</span>`, 'pill', [], `${L('已替换', 'Replaced')} ${units(o.n)}`];
    case 'copied': return [`${CHECK}<span class="lbl">${L('已复制到剪贴板', 'Copied to the clipboard')}</span>`, 'pill', [], L('已复制到剪贴板', 'Copied to the clipboard')];
    case 'perm': return [`<span class="lead warn">${I('a11y')}</span><span class="lbl">${L('需要辅助功能权限', 'Needs Accessibility permission')} <span class="sub">· ${L('否则按键传不到言字', 'or Yana never hears the key')}</span></span>`, 'pill', [BUD.perm()], L('需要辅助功能权限', 'Needs Accessibility permission')];
    case 'ready': return [`${CHECK}<span class="lbl">${L(`${esc(o.name || '言字')}准备好了`, `${esc(o.name || 'Yana')} is ready`)} <span class="sub">· ${L('点一下右 ⌘ 试试', 'tap right ⌘ to try it')}</span></span>`, 'pill', [], L('准备好了，点一下右 ⌘ 试试', 'Ready: tap right ⌘ to try it')];
    case 'notice': return [`<span class="lead">${I('info')}</span><span class="lbl">${esc(o.msg || '')}</span>`, 'pill', [], o.msg || ''];
    case 'error': return [`<span class="lead err wiggle">${I('warn')}</span><span class="lbl">${L('出错了', 'Something went wrong')}${o.msg ? ` <span class="sub">· ${esc(o.msg)}</span>` : ''}</span>`, 'pill', o.log === false ? [] : [BUD.log()], L('出错了', 'Something went wrong') + (o.msg ? L('，', ': ') + o.msg : '')];
  }
  return ['', 'pill', [], ''];
}

/* ---------------- layout ---------------- */
function anchorPoint() {
  const W = innerWidth, H = innerHeight, a = S.anchor;
  return { x: a.x == null ? W / 2 : a.x, y: a.y == null ? (a.mode === 'top' ? MARGIN : H - MARGIN) : a.y, top: a.mode === 'top' };
}
function layout(capW, capH, budWs) {
  const W = innerWidth, ap = anchorPoint();
  const total = capW + budWs.reduce((sum, w) => sum + GAP + w, 0);
  const left = Math.max(8, Math.min(W - 8 - total, ap.x - total / 2));
  const top = ap.top ? ap.y : ap.y - capH;
  const out = { cap: { x: left, y: top, w: capW, h: capH, r: capH > PILL_H ? 24 : 20 }, buds: [] };
  let x = left + capW + GAP;
  for (const w of budWs) { out.buds.push({ x, y: top + (PILL_H - BUD_H) / 2, w, h: BUD_H, r: BUD_H / 2 }); x += w + GAP; }
  return out;
}

/* ---------------- render ---------------- */
function measureLayer(html, kind) {
  const el = document.createElement('div');
  el.className = `layer ${kind === 'card' ? 'card ' : ''}measure enter`;
  if (kind === 'card') el.style.width = Math.min(400, innerWidth - 16) + 'px';
  el.innerHTML = html;
  capEl.appendChild(el);
  el.querySelectorAll('.en-t').forEach(enFade);
  const w = Math.ceil(el.offsetWidth), h = Math.ceil(el.offsetHeight);
  el.classList.remove('measure');
  return { el, w, h };
}
function makeBuds(list) {
  return list.map(html => {
    const t = document.createElement('div');
    t.innerHTML = html;
    const b = t.firstElementChild;
    b.classList.add('bud', 'gl', 'measure');
    budsEl.appendChild(b);
    const w = Math.ceil(b.offsetWidth);
    b.classList.remove('measure');
    return { el: b, w };
  });
}
function dropBuds(now) {
  for (const b of buds) {
    if (b.gone) continue;
    b.gone = true;
    b.el.style.pointerEvents = 'none';
    b.o.animate(0, 'dismiss', now); b.s.animate(.7, 'dismiss', now); b.tx.animate(-18, 'dismiss', now);
  }
}

function render(st, d) {
  const now = performance.now();
  const [html, kind, budList, say] = view(st, d);
  const fromNothing = S.hidden && cap.o.value(now) < .05;
  const { el, w, h } = measureLayer(html, kind);
  const capW = kind === 'circle' ? 40 : Math.max(40, w);
  const capH = kind === 'card' ? h : PILL_H;
  const made = makeBuds(budList);
  const L = layout(capW, capH, made.map(b => b.w));
  const spring = (kind === 'card') !== (S.kind === 'card') ? 'card' : 'morph';

  if (fromNothing) {
    cap.place({ x: L.cap.x + L.cap.w / 2 - 20, y: anchorPoint().top ? L.cap.y : L.cap.y + L.cap.h - PILL_H, w: 40, h: PILL_H, r: 20 });
    cap.s.set(.82); cap.o.set(0); cap.ty.set(10); cap.tx.set(0);
  }
  cap.go(L.cap, spring, now);
  cap.s.animate(1, 'appear', now); cap.o.animate(1, 'appear', now); cap.ty.animate(0, 'appear', now);
  cap.gone = false;
  capEl.style.pointerEvents = '';

  void el.offsetWidth;
  el.classList.remove('enter');
  if (S.layer) { const old = S.layer; old.classList.add('leave'); setTimeout(() => old.remove(), 220); }
  S.layer = el;

  dropBuds(now);
  made.forEach((m, i) => {
    const b = new Shape('b' + (++seq), m.el);
    const r = L.buds[i];
    b.place(r);
    b.o.set(0); b.s.set(.55); b.tx.set(-26);
    const delay = (fromNothing ? .2 : .09) + i * .06;
    b.o.animate(1, 'bud', now, delay); b.s.animate(1, 'bud', now, delay); b.tx.animate(0, 'bud', now, delay);
    buds.push(b);
  });

  S.st = st; S.kind = kind; S.hidden = false;
  handleEl.classList.add('away');
  S.last = { capW, capH, budWs: made.map(m => m.w) };
  capEl.setAttribute('aria-hidden', 'false');
  if (say && say !== S.say) { S.say = say; post({ t: 'say', text: say }); }
  if (kind === 'card') setTimeout(focusField, 60);
  kick();
}

// Progress arrives every half second; redrawing the whole capsule each time
// would cross-fade it constantly, so only the numbers change.
function updateDownload(d) {
  S.d = d;
  const p = Math.max(0, Math.min(1, +d.p || 0));
  const ring = S.layer.querySelector('.pring'), pct = S.layer.querySelector('.pct'), eta = S.layer.querySelector('.eta');
  if (ring) ring.style.setProperty('--p', p.toFixed(3));
  if (pct) pct.textContent = Math.round(p * 100) + '%';
  if (eta) eta.textContent = d.eta ? ` · ${d.eta}` : '';
  const w = Math.max(40, Math.ceil(S.layer.offsetWidth));
  if (S.last && Math.abs(w - S.last.capW) > 1) { S.last.capW = w; relayout(); }
  const say = downloadSay(p);
  if (say !== S.say) { S.say = say; post({ t: 'say', text: say }); }
}

function relayout() {
  if (S.hidden || !S.last) return;
  const now = performance.now();
  const L = layout(S.last.capW, S.last.capH, S.last.budWs);
  cap.go(L.cap, 'morph', now);
  buds.filter(b => !b.gone).forEach((b, i) => { if (L.buds[i]) b.go(L.buds[i], 'morph', now); });
  kick();
}

function hide() {
  const now = performance.now();
  S.hidden = true; S.st = 'hidden'; S.say = '';
  handleEl.classList.remove('away');
  cap.s.animate(.82, 'dismiss', now); cap.o.animate(0, 'dismiss', now); cap.ty.animate(10, 'dismiss', now);
  cap.gone = true;
  capEl.style.pointerEvents = 'none';
  capEl.setAttribute('aria-hidden', 'true');
  dropBuds(now);
  const field = $('#cardField');
  if (field) field.blur();
  kick();
}

function focusField() {
  const f = $('#cardField');
  if (!f || S.kind !== 'card') return;
  f.focus();
  f.setSelectionRange(f.value.length, f.value.length);
}

/* ---------------- the recording view: timer, countdown, quiet, wave ---------------- */
const hist = new Array(8).fill(0), bh = new Array(NB).fill(3);
function rerenderRec() { if (S.st === 'rec') render('rec', S.d); }
function tickRec(now) {
  if (S.st !== 'rec' || !S.layer) return;
  const t = (now - S.recAt) / 1000;
  const rem = S.max - t;
  if (S.max > 0 && rem <= 60 && !S.cd) { S.cd = true; rerenderRec(); return; }
  if (S.mode !== 'hold' && !S.cd) {
    // Only a microphone that has picked up nothing at all is worth a word:
    // once the wave has moved, a pause is someone thinking, not a fault.
    const quiet = !S.heard && now - S.lastVoice > QUIET_MS;
    if (quiet !== S.quiet) { S.quiet = quiet; rerenderRec(); return; }
  }
  const txt = S.cd ? fmt(rem, true) : fmt(t);
  if (txt !== S.tmrText) {
    S.tmrText = txt;
    const el = S.layer.querySelector('.tmr');
    if (el) el.textContent = txt;
  }
  if (S.cd) {
    const fg = S.layer.querySelector('.ring .fg');
    if (fg) fg.setAttribute('stroke-dashoffset', (103.7 * (1 - Math.max(0, Math.min(1, rem / 60)))).toFixed(1));
  }
  if (now - S.levelAt > 150) S.level *= .85;
  hist.unshift(S.level); hist.length = 8;
  const c = (NB - 1) / 2, still = reduceMotion();
  S.layer.querySelectorAll('.wave i').forEach((b, i) => {
    const dist = Math.min(7, Math.round(Math.abs(i - c)));
    const target = 3 + 21 * hist[dist] * PROF[dist] * (still ? .9 : (.72 + Math.random() * .4));
    bh[i] += (target - bh[i]) * (target > bh[i] ? .55 : .2);
    b.style.height = Math.max(3, Math.min(24, bh[i])).toFixed(1) + 'px';
  });
}

/* ---------------- idle handle ---------------- */
function handleRects() {
  const ap = anchorPoint();
  const bottom = ap.top ? innerHeight - MARGIN : ap.y;
  return { line: { x: ap.x - 22, y: bottom - 18, w: 44, h: 18 }, button: { x: ap.x - 20, y: bottom - 40, w: 40, h: 40, r: 20 } };
}
function setHandle(onOff) {
  S.handleOn = !!onOff;
  handleEl.hidden = !S.handleOn;
  hbEl.hidden = !S.handleOn;
  if (!S.handleOn) setHandleHover(false);
  placeHandle();
  kick();
}
function placeHandle() {
  const R = handleRects();
  handleEl.style.transform = `translate(${R.line.x}px, ${R.line.y}px)`;
  hb.place(R.button);
}
function setHandleHover(v) {
  if (v === S.handleHover) return;
  S.handleHover = v;
  handleEl.classList.toggle('hov', v);
  hbEl.style.pointerEvents = v ? 'auto' : 'none';
  hbEl.innerHTML = v ? I('mic') : '';
  const now = performance.now();
  if (v) { hb.s.set(.4); hb.o.animate(1, 'appear', now); hb.s.animate(1, 'appear', now); }
  else { hb.o.animate(0, 'dismiss', now); hb.s.animate(.4, 'dismiss', now); }
  kick();
}

/* ---------------- frame loop ---------------- */
let raf = 0;
function kick() { if (!raf) raf = requestAnimationFrame(loop); }
function loop(now) {
  raf = 0;
  const out = [];
  let busy = false;
  const cf = cap.apply(now);
  if (cf.o > .001 || cap.busy()) out.push(cap.geo(cf));
  busy = busy || cap.busy();
  buds = buds.filter(b => {
    const f = b.apply(now);
    const moving = b.busy();
    if (b.gone && !moving) { b.el.remove(); return false; }
    out.push(b.geo(f));
    busy = busy || moving;
    return true;
  });
  if (S.handleOn) {
    const R = handleRects();
    out.push({ id: 'handle', x: R.line.x, y: R.line.y, w: R.line.w, h: R.line.h, r: 0, a: S.hidden ? 1 : 0, glass: false, hit: S.hidden });
    const hf = hb.apply(now);
    if (hf.o > .001) out.push(hb.geo(hf));
    busy = busy || hb.busy();
  }
  if (S.hidden && !cap.busy() && S.layer) { S.layer.remove(); S.layer = null; }
  const g = JSON.stringify(out);
  if (g !== S.lastGeo) { S.lastGeo = g; post({ t: 'geo', s: out }); }
  tickRec(now);
  if (busy || S.st === 'rec') kick();
}

/* ---------------- input ---------------- */
function act(a) {
  const msg = { t: 'act', a };
  if (a === 'useen' || a === 'copyen') msg.text = S.d.en || '';
  if (a === 'done' || a === 'replace') { const f = $('#cardField'); msg.text = f ? f.value : ''; }
  post(msg);
}
// Drag across words of the sentence to save them as one phrase; a press and release on one word saves that word.
let drag = null;
const save = (term, kind, on) => post({ t: 'act', a: 'word', term, kind, on });
function dragTo(el) {
  const w = el && el.closest && el.closest('.en-t .wd');
  if (!drag || !w || w === drag.cur) return;
  drag.cur = w;
  const ws = [...drag.t.querySelectorAll('.wd')], kids = [...drag.t.children];
  const a = ws.indexOf(drag.from), b = ws.indexOf(w);
  const lo = kids.indexOf(ws[Math.min(a, b)]), hi = kids.indexOf(ws[Math.max(a, b)]);
  kids.forEach((k, i) => k.classList.toggle('sel', i >= lo && i <= hi));
  drag.n = Math.abs(a - b) + 1;
}
function dragEnd() {
  const d = drag; drag = null;
  if (!d) return;
  d.t.classList.remove('dragging');
  const sel = [...d.t.querySelectorAll('.sel')];
  sel.forEach(k => k.classList.remove('sel'));
  if (d.n >= 2) {
    const ws = sel.filter(k => k.classList.contains('wd')), last = ws[ws.length - 1];
    const term = S.d.en.slice(+ws[0].dataset.i, +last.dataset.i + last.textContent.length);
    sel.forEach(k => (k._b = k._b || []).push(term) && k.classList.add('saved'));
    save(term, 'phrase', true);
    return;
  }
  const w = d.from, b = w._b && w._b.pop();
  if (b) {   // inside a saved block: take the whole block back
    document.querySelectorAll('.en-t .saved').forEach(k => {
      const i = (k._b || []).indexOf(b);
      if (i >= 0) k._b.splice(i, 1);
      if (!(k._b || []).length) k.classList.remove('saved');
    });
    save(b, w.dataset.k, false);
  } else {
    const on = w.classList.toggle('saved');
    if (on) (w._b = w._b || []).push(w.dataset.w);
    save(w.dataset.w, 'word', on);
  }
}
stage.addEventListener('mousedown', e => {
  const w = e.target.closest('.en-t .wd');
  if (!w || e.button) return;
  e.preventDefault();
  drag = { from: w, cur: null, n: 1, t: w.closest('.en-t') };
  drag.t.classList.add('dragging');
  dragTo(w);
});
document.addEventListener('mousemove', e => { if (drag) dragTo(e.target); });
document.addEventListener('mouseup', dragEnd);
stage.addEventListener('click', e => {
  const w = e.target.closest('[data-w]');
  if (w) {
    if (w.closest('.en-t')) return;
    const on = w.classList.toggle('saved');
    save(w.dataset.w, w.dataset.k, on);
    return;
  }
  const b = e.target.closest('[data-act]');
  if (b) act(b.dataset.act);
});
capEl.addEventListener('keydown', e => {
  if (e.target.id !== 'cardField') return;
  if (e.isComposing || e.keyCode === 229) return;
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const b = capEl.querySelector('.layer:not(.leave) [data-act="replace"], .layer:not(.leave) [data-act="done"]');
    if (b) act(b.dataset.act);
  } else if (e.key === 'Escape') { e.preventDefault(); act('close'); }
});
let liveT = 0;
capEl.addEventListener('input', e => {
  if (e.target.id !== 'cardField') return;
  clearTimeout(liveT);
  const v = e.target.value;
  liveT = setTimeout(() => post({ t: 'edit', text: v }), 250);
});
capEl.addEventListener('focusin', e => { if (e.target.id === 'cardField') post({ t: 'field', focus: true }); });
capEl.addEventListener('focusout', e => { if (e.target.id === 'cardField') post({ t: 'field', focus: false }); });

// The panel never becomes key for plain clicks, so WebKit gets no hover;
// Python polls the pointer and hands it over.
function pointer(x, y) {
  let over = null;
  if (x != null) over = document.elementFromPoint(x, y);
  if (drag) dragTo(over);
  const btn = over ? over.closest('button, .wd') : null;
  if (btn !== S.hovBtn) {
    if (S.hovBtn) S.hovBtn.classList.remove('hov');
    if (btn) btn.classList.add('hov');
    S.hovBtn = btn;
  }
  const onCap = !!(over && !S.hidden && (over.closest('#cap') || over.closest('.bud')));
  if (onCap !== S.hover) { S.hover = onCap; post({ t: 'hover', on: onCap }); }
  if (S.handleOn) setHandleHover(!!(over && (over.closest('#handle') || over.closest('#hb'))) && S.hidden);
}

/* ---------------- messages from Python ---------------- */
on('show', m => {
  const d = m.d || {};
  if (m.st === 'rec') {
    const now = performance.now();
    const entering = S.st !== 'rec';
    if (d.elapsed != null) S.recAt = now - (+d.elapsed) * 1000;
    else if (entering) S.recAt = now;
    if (entering) { S.lastVoice = now; S.quiet = false; S.heard = false; S.cd = false; }
    S.mode = d.mode || 'click';
    if (d.max != null) S.max = +d.max;
    // Extended: back from the countdown to the running timer.
    if (S.cd && S.max - (now - S.recAt) / 1000 > 60) S.cd = false;
  }
  if (m.st === 'download' && S.st === 'download' && !S.hidden && S.layer) { updateDownload(d); return; }
  S.d = d;
  render(m.st, d);
});
on('hide', hide);
on('level', m => {
  S.level = Math.max(0, Math.min(1, +m.v || 0));
  S.levelAt = performance.now();
  if (S.level > VOICE) {
    S.lastVoice = S.levelAt;
    S.heard = true;
    if (S.quiet && S.st === 'rec') { S.quiet = false; rerenderRec(); }
  }
});
on('pointer', m => pointer(m.x, m.y));
on('env', m => {
  const relabel = env(m);
  S.native = !!m.native;
  document.documentElement.classList.toggle('native', S.native);
  hbEl.setAttribute('aria-label', L('开始听写', 'Start dictation'));
  document.title = L('言字', 'Yana');
  if (relabel && !S.hidden) { S.say = ''; render(S.st, S.d); }
});
on('anchor', m => { S.anchor = { mode: m.mode || 'bottom', x: m.x == null ? null : +m.x, y: m.y == null ? null : +m.y }; placeHandle(); relayout(); });
on('handle', m => setHandle(m.on));
on('focus', focusField);
addEventListener('resize', () => { placeHandle(); relayout(); });

window.app.debug = { S, view, render, hide, layout };
post({ t: 'ready' });
})();
