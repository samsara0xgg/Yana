/* Settings. Python sends the whole state ({t:'state'}) whenever it changes;
   this page only draws it and reports what the user changed. Nothing is kept
   here that Python does not also know, except what is being typed. */
(() => {
'use strict';
const { I, esc, $, $$, post, on, env, L } = kit;

// [id, [Chinese, English] title, icon, tile colour]
const PANES = [
  ['general', ['通用', 'General'], 'gear', '#8E8E93'],
  ['dictation', ['听写', 'Dictation'], 'wave', '#007AFF'],
  ['keys', ['快捷键', 'Shortcuts'], 'keyboard', '#636366'],
  ['model', ['润色', 'Refinement'], 'sparkles', '#AF52DE'],
  ['vocab', ['词库', 'Vocabulary'], 'book', '#FF9500'],
  ['audio', ['音频', 'Audio'], 'speaker', '#FF3B30'],
  ['usage', ['用量', 'Usage'], 'chart', '#34C759'],
  ['privacy', ['历史与隐私', 'History & Privacy'], 'shield', '#0A84FF'],
  ['about', ['关于', 'About'], 'info', '#5AC8FA'],
];
const paneTitle = id => L(...PANES.find(p => p[0] === id)[1]);
const DAYS = v => ({ 30: L('30 天', '30 days'), 90: L('90 天', '90 days'), 365: L('1 年', '1 year'), 0: L('永久', 'Forever') })[v] || L(`${v} 天`, `${v} days`);
const LANGS = () => [['', L('自动', 'Automatic')], ['zh', '中文'], ['en', 'English'], ['fr', 'Français']];
// The interface language names itself in both, so it can be found whichever one is showing.
const UI_LANGS = () => [['auto', L('跟随系统', 'Same as the Mac')], ['zh', '中文'], ['en', 'English']];
const SOURCES = () => [['auto', L('自动', 'Automatic')], ['huggingface', 'Hugging Face'], ['mirror', L('国内镜像 hf-mirror.com', 'China mirror (hf-mirror.com)')]];

let S = null;            // the state Python sent
let pane = 'general';
// What is being edited here and not yet sent.
const UI = { keyEdit: '', keyBusy: '', keyMsg: {}, test: null, confirm: null, vocabMsg: '', usageBy: '',
  // The feedback box: kept here so a redraw or a failed send never loses what was typed.
  fb: { text: '', email: '', diag: false, status: '', msg: '' } };

/* ---------------- building blocks ---------------- */
const sw = (key, on_, label, disabled = false) =>
  `<button class="sw" role="switch" aria-checked="${!!on_}" aria-label="${esc(label)}" data-set="${key}"${disabled ? ' disabled' : ''}></button>`;
const select = (key, value, options, label) =>
  `<select class="pop" data-set="${key}" aria-label="${esc(label)}">${options.map(([v, t]) =>
    `<option value="${esc(v)}"${String(v) === String(value) ? ' selected' : ''}>${esc(t)}</option>`).join('')}</select>`;
const seg = (key, value, options) =>
  `<span class="msg" role="radiogroup">${options.map(([v, t]) =>
    `<button role="radio" aria-checked="${String(v) === String(value)}" data-set="${key}" data-v="${esc(v)}">${esc(t)}</button>`).join('')}</span>`;
const row = (label, control, note = '', cls = '') =>
  `<div class="row ${cls}"><div class="rl">${label}${note ? `<small>${note}</small>` : ''}</div>${control}</div>`;
const grp = (rows, extra = '') => `<div class="grp">${rows.join('')}${extra}</div>`;
const head = t => `<div class="pane-h"><b>${esc(t)}</b></div>`;
const secs = v => L(`${v} 秒`, `${v} s`);
const times = n => L(`${n} 次`, `${n} dictation${n === 1 ? '' : 's'}`);

/* ---------------- panes ---------------- */
function general() {
  const p = S.prefs, login = S.login;
  const loginNote = login === 'unavailable' ? L('只有安装好的 App 才能设置，从源码运行时不可用。', 'Only the installed app can do this; not available when running from source.')
    : login === 'requires_approval' ? L('需要在「系统设置 › 通用 › 登录项」里允许。', 'Allow it in System Settings › General › Login Items.') : '';
  const loginLabel = L('登录时打开', 'Open at login');
  const loginCtl = login === 'requires_approval'
    ? `<span class="inline">${sw('login', true, loginLabel)}<button class="mbtn" data-act="open" data-what="login">${L('打开登录项…', 'Open Login Items…')}</button></span>`
    : sw('login', login === 'enabled', loginLabel, login === 'unavailable');
  const langLabel = L('界面语言 · Language', 'Language · 界面语言');
  return head(paneTitle('general'))
    + grp([
      row(esc(langLabel), select('ui_language', p.ui_language, UI_LANGS(), langLabel), L('菜单、胶囊和所有窗口的文字。', 'The words in the menu, the capsule and every window.')),
    ])
    + grp([
      row(L(`登录时打开${esc(S.name)}`, `Open ${esc(S.name)} at login`), loginCtl, loginNote),
      row(L('胶囊位置', 'Capsule position'), seg('capsule_position', p.capsule_position, [['bottom', L('屏幕底部', 'Bottom of screen')], ['caret', L('跟随光标', 'At the cursor')]]),
        L('跟随光标需要目标 App 提供光标位置；拿不到时回到屏幕底部。', 'At the cursor needs the app to report where its cursor is; otherwise it goes back to the bottom.')),
      row(L('显示底部把手', 'Show the handle'), sw('show_handle', p.show_handle, L('显示底部把手', 'Show the handle')),
        L('空闲时在 Dock 上方留一道细线，悬停变成麦克风按钮。', 'A thin line above the Dock while idle; hover to turn it into a microphone button.')),
      row(L('结果停留', 'Result stays for'), select('dismiss_seconds', p.dismiss_seconds, S.choices.dismiss_seconds.map(v => [v, secs(v)]), L('结果停留', 'Result stays for')),
        L('指针停在胶囊上时不计时；接着打字会立刻收起。', 'The pointer resting on the capsule pauses it; typing dismisses it at once.')),
    ])
    + grp([row(L('提示音', 'Sounds'), select('sounds', p.sounds, [['off', L('关闭', 'Off')], ['start_end', L('仅开始和结束', 'Start and end only')]], L('提示音', 'Sounds')))])
    + grp([row(L('诊断信息', 'Diagnostics'), `<button class="mbtn" data-act="open" data-what="diagnostics">${L('导出…', 'Export…')}</button>`,
      L('反馈问题时附上这个文件：日志、改过的设置和这台 Mac 的情况。不含 API Key，也不含听写历史。',
        'Attach this file when you report a problem: the log, the settings you changed and details of this Mac. No API keys, no dictation history.'))]);
}

function dictation() {
  const p = S.prefs, cloud = p.asr_engine === 'qwen';
  return head(paneTitle('dictation'))
    + grp([
      row(L('识别语言', 'Spoken language'), select('language', S.language, LANGS(), L('识别语言', 'Spoken language')),
        L('自动会听出你说的语言，混着说也行。只说一种语言时才固定它，否则别的语言可能被认错或翻译。', 'Automatic hears which language you speak, even mixed. Fix one only if it is the only language you speak; others may be misheard or translated.')),
      row(L('语音识别', 'Speech recognition'), seg('asr_engine', p.asr_engine, [['local', L('这台 Mac', 'This Mac')], ['qwen', L('千问（云端）', 'Qwen (cloud)')]]),
        cloud ? L('边说边传到阿里云新加坡识别，松手后不到 1 秒出字。没有 Key、断网或出错时自动用这台 Mac 上的模型。想要原话直出，把下面的润色关掉。',
          'Streams to Alibaba Cloud in Singapore while you talk; text arrives under a second after you finish. With no key, no network or an error, the model on this Mac takes over. For your exact words, turn Refine off below.')
          : L('Whisper 在这台 Mac 上识别，音频不离开电脑。', 'Whisper recognizes speech on this Mac; audio never leaves it.')),
      ...(cloud ? [keyRow(S.asrKey)] : []),
      row(L('模型下载源', 'Model download'), select('model_source', p.model_source, SOURCES(), L('模型下载源', 'Model download')),
        L('自动：在中国大陆用国内镜像，其他地方用 Hugging Face。只管下载，识别始终在这台 Mac 上。',
          'Automatic uses the China mirror in mainland China and Hugging Face elsewhere. Only for the download; recognition always runs on this Mac.')),
      row(L('单次最长录音', 'Longest recording'), select('max_minutes', p.max_minutes, S.choices.max_minutes.map(v => [v, L(`${v} 分钟`, `${v} minutes`)]), L('单次最长录音', 'Longest recording')),
        L('最后 60 秒会倒计时。', 'The last 60 seconds count down.')),
    ])
    + grp([
      row(L('润色', 'Refine'), sw('refine', p.refine, L('润色', 'Refine')), L('去掉口头禅、处理改口、补标点和分段。', 'Removes filler words, applies self-corrections, adds punctuation and paragraphs.')),
      row(L('润色超时或失败时', 'If refinement fails'), `<span class="val">${L('插入原始转写', 'Insert the raw transcript')}</span>`, L('胶囊里可以一键重新润色。', 'The capsule offers to refine it again.')),
      row(L('改写所选文字', 'Rewrite the selection'), sw('rewrite_selection', p.rewrite_selection, L('改写所选文字', 'Rewrite the selection')),
        L('先选中文字再轻点右 ⌘，说出要求，例如“改得更正式”“翻成英文”。', 'Select text, tap right ⌘ and say what to do, like “make it more formal” or “translate to Chinese”.')),
      row(L('没有输入框时', 'With no text field'), `<span class="val">${L('显示卡片并复制', 'Show a card and copy')}</span>`,
        L('结果放进可以修改的卡片，同时复制到剪贴板。', 'The text goes into a card you can edit, and onto the clipboard.')),
    ]);
}

function keys() {
  const f5 = S.prefs.f5_hotkey, rc = S.prefs.right_command_hotkey !== false;
  const top = ['esc', 'F1', 'F2', 'F3', 'F4', '', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'];
  const kb = `<div class="kb" aria-hidden="true">${top.map(k => k ? `<i>${k}</i>` : `<i${f5 ? ' class="hot"' : ''}>${I('mic')}</i>`).join('')}<i class="w2"></i>`
    + `<i>fn</i><i>⌃</i><i>⌥</i><i class="w2">⌘</i><i class="w6">space</i><i class="w2${rc ? ' hot' : ''}">⌘</i><i>⌥</i><i>←→</i></div>`;
  const conflict = f5 && S.f5 ? row(L(`系统听写也在用 ${I('mic')} 键`, `System dictation also uses the ${I('mic')} key`),
    `<button class="mbtn" data-act="open" data-what="keyboard">${L('打开键盘设置…', 'Open Keyboard Settings…')}</button>`,
    L('两个都开会互相抢。在键盘设置里把系统听写的快捷键换掉或关闭，或者关掉上面这个开关。',
      'The two fight over it. Change or turn off the system dictation shortcut in Keyboard settings, or turn this switch off.'), 'warnrow') : '';
  const key = [rc && `<kbd>${L('右 ⌘', 'Right ⌘')}</kbd>`, f5 && '<kbd>F5</kbd>'].filter(Boolean).join(`<span class="arrow">${L('或', 'or')}</span>`);
  // Right ⌘ + Space only locks a recording already running (from idle it stays Spotlight), so it is a note, not a shortcut here.
  const lock = rc ? '' : `<span class="arrow">·</span><kbd>F5</kbd><kbd>Space</kbd>`;
  return head(paneTitle('keys')) + kb
    + grp([
      row(L('开始 / 结束', 'Start / finish'), `<span class="kbd2"><span class="arrow">${L('轻点', 'tap')}</span>${key}</span>`),
      row(L('按住说话', 'Hold to talk'), `<span class="kbd2"><span class="arrow">${L('按住', 'hold')}</span>${key}</span>`, L('按住一会儿就开始录，松开就完成。', 'Recording starts after a moment; letting go finishes.')),
      row(L('锁定（免手持）', 'Lock (hands-free)'), `<span class="kbd2"><span class="arrow">${L('连点两下', 'double-tap')}</span>${key}${lock}</span>`,
        rc ? L('录音时按右 ⌘ + 空格也能锁定。', 'While recording, right ⌘ + Space also locks it.') : ''),
      row(L('取消', 'Cancel'), '<span class="kbd2"><kbd>esc</kbd></span>'),
    ])
    + grp([
      row(L('用右 ⌘ 听写', 'Dictate with right ⌘'), sw('right_command_hotkey', rc, L('用右 ⌘ 听写', 'Dictate with right ⌘'), rc && !f5),
        L('如果你的右 ⌘ 已经用来切换输入法，关掉它，改用 F5。', 'If right ⌘ already switches your input source, turn this off and use F5.')),
      row(L(`也用 F5（${I('mic')} 键）`, `Also use F5 (the ${I('mic')} key)`), sw('f5_hotkey', f5, L('也用 F5', 'Also use F5'), f5 && !rc),
        L(`MacBook 上 F5 是系统听写键。打开后${esc(S.name)}会接管它，系统听写就不能用这个键了。`, 'On a MacBook F5 is the system dictation key. When this is on, the app takes it over and system dictation can no longer use it.')),
    ], conflict);
}

function model() {
  // One model for everyone, never named: show whose key pays and how fast it is. Nothing to pick.
  const m = S.presets.find(p => p.name === S.active) || {};
  const how = !S.prefs.refine ? L('关闭', 'Off') : S.trial.on ? L('免费试用', 'Free trial') : L(`你的 ${m.service || 'OpenAI'} Key`, `Your ${m.service || 'OpenAI'} key`);
  const modelRow = row(L('润色', 'Refinement'), `<span class="val">${esc(how)}</span>`,
    m.median ? L(`中位延迟 ${(m.median / 1000).toFixed(1)} 秒`, `Median time ${(m.median / 1000).toFixed(1)} s`) : '');
  const t = UI.test;
  const testLine = !t ? '' : t.busy ? `<span class="val"><span class="spin"></span> ${L('正在发送一次测试请求…', 'Sending a test request…')}</span>`
    : t.ok ? `<span class="st-ok">${L('已连接', 'Connected')} · ${L(`往返 ${(t.ms / 1000).toFixed(1)} 秒`, `${(t.ms / 1000).toFixed(1)} s round trip`)}</span>`
    : `<span class="st-err">${esc(t.msg || L('连接失败', 'Connection failed'))}</span>`;
  const keyRows = S.keys.map(keyRow);
  const envKeys = S.keys.filter(k => k.where === 'env');
  const migrate = envKeys.length
    ? `<p class="note">${envKeys.map(k => esc(k.env)).join(L('、', ', '))} ${L('还以明文存在旧版的配置文件里。', 'is still in plain text in an old settings file.')}<button class="mbtn" data-act="migrate">${L('移到钥匙串', 'Move to Keychain')}</button></p>` : '';
  const trialNote = !S.trial.on ? '' : `<p class="note">${S.trial.over
      ? L('免费试用已经用完，现在只插入原始转写。填上你自己的 OpenAI API Key 就能继续润色。', 'The free trial is used up, so dictation inserts the raw transcript. Add your own OpenAI API key to keep refining.')
      : L('正在用免费试用：前大约 300 次润色免费。想长期用，填上你自己的 OpenAI API Key，会自动切换过去。', 'You are on the free trial: about the first 300 refinements are free. For the long run, add your own OpenAI API key and it switches over by itself.')}
    <button class="mbtn" data-act="open" data-what="openai-keys">${L('去 OpenAI 申请 Key', 'Get a Key from OpenAI')}</button></p>`;
  return head(paneTitle('model')) + trialNote
    + (S.prefs.refine ? '' : `<p class="note">${L('润色已关闭，听写会直接插入原始转写。可以在「听写」里打开。', 'Refinement is off, so dictation inserts the raw transcript. Turn it on under Dictation.')}</p>`)
    + `<div class="grp">${modelRow}<div class="btnrow">${testLine}<button class="mbtn" data-act="test"${t && t.busy ? ' disabled' : ''}>${L('测试连接', 'Test Connection')}</button></div></div>`
    + `<div class="grp-l">${L('API Key · 保存在钥匙串', 'API key · kept in the keychain')}</div>` + grp(keyRows) + migrate;
}

function keyRow(k) {
  const where = { keychain: L('已保存', 'Saved'), env: L('明文保存在 env 文件', 'In plain text in the env file'), environ: L('来自环境变量', 'From the environment'), '': L('未设置', 'Not set') }[k.where] || '';
  const hint = [k.env, k.hint || where].filter(Boolean).join(' · ');
  const msg = UI.keyMsg[k.env];
  const msgLine = msg ? `<small class="${msg.ok ? 'st-ok' : 'st-err'}">${esc(msg.text)}</small>` : '';
  if (UI.keyEdit === k.env) {
    const busy = UI.keyBusy === k.env;
    return `<div class="row"><div class="rl">${esc(k.service)}<small>${esc(k.env)}</small>${msgLine}</div>
      <span class="inline"><input class="field" id="key-${esc(k.env)}" type="password" placeholder="${L('粘贴 API Key', 'Paste the API key')}" autocomplete="off" spellcheck="false" style="width:220px" aria-label="${esc(k.service)} API Key">
      <button class="mbtn pri" data-act="savekey" data-env="${esc(k.env)}"${busy ? ' disabled' : ''}>${busy ? '<span class="spin"></span>' : L('保存', 'Save')}</button>
      <button class="mbtn" data-act="cancelkey">${L('取消', 'Cancel')}</button></span></div>`;
  }
  return `<div class="row"><div class="rl">${esc(k.service)}<small>${esc(hint)}</small>${msgLine}</div>
    <button class="mbtn${k.where ? '' : ' pri'}" data-act="editkey" data-env="${esc(k.env)}">${k.where ? L('更换…', 'Change…') : L('添加…', 'Add…')}</button></div>`;
}

function vocab() {
  const v = S.vocab;
  const mine = v.mine.map(w => `<span class="wchip">${esc(w)}<button data-act="unword" data-term="${esc(w)}" aria-label="${L('删除', 'Remove')} ${esc(w)}">${I('xmark')}</button></span>`).join('');
  const suggest = v.suggest.length ? v.suggest.map(s => row(`${esc(s.term)}`,
    `<span class="inline"><button class="mbtn" data-act="reject" data-term="${esc(s.term)}">${L('忽略', 'Ignore')}</button><button class="mbtn pri" data-act="word" data-term="${esc(s.term)}">${L('加入', 'Add')}</button></span>`,
    s.n ? L(`最近 30 天出现 ${s.n} 次`, `${s.n} time${s.n === 1 ? '' : 's'} in the last 30 days`) : ''))
    : [row(`<span class="empty">${L('暂时没有建议。多听写几天，这里会列出常被写错的词。', 'No suggestions yet. After a few days of dictation, words that keep coming out wrong show up here.')}</span>`, '')];
  const fixes = v.fixes.length ? v.fixes.map(f => row(`${f.wrong ? `<span class="del">${esc(f.wrong)}</span> <span class="arrow">→</span> ` : ''}${esc(f.right)}`,
    v.mine.includes(f.right) ? `<span class="val">${L('已在词库', 'In the vocabulary')}</span>` : `<button class="mbtn" data-act="word" data-term="${esc(f.right)}">${L('加入词库', 'Add to Vocabulary')}</button>`, esc(f.at || '')))
    : [row(`<span class="empty">${L('在胶囊的修改卡片里改过的词会出现在这里。', 'Words you fix in the capsule’s edit card show up here.')}</span>`, '')];
  const learned = v.learned.map(f => row(`${f.wrong ? `<span class="del">${esc(f.wrong)}</span> <span class="arrow">→</span> ` : ''}${esc(f.right)}${f.n > 1 ? ` <span class="val">×${f.n}</span>` : ''}`,
    `<button class="mbtn" data-act="unlearn" data-term="${esc(f.right)}">${L('移除', 'Remove')}</button>`, esc(f.at || '')));
  return head(paneTitle('vocab'))
    + `<div class="grp-l">${L('我的词 · 同时提示给语音识别和润色模型', 'My words · hinted to both speech recognition and refinement')}</div>`
    + `<div class="grp"><div class="chipset">${mine}<input class="field" id="newword" placeholder="${L('添加词，回车确认', 'Add a word, press Return')}" aria-label="${L('添加词', 'Add a word')}"></div></div>`
    + (UI.vocabMsg ? `<p class="note">${esc(UI.vocabMsg)}</p>` : '')
    + `<div class="grp-l">${L('自动学到的 · 来自你发送前的修改', 'Learned from your edits')}</div>`
    + grp(learned.length ? learned : [row(`<span class="empty">${L('在输入框里把听错的词改对再发送，改对的词会出现在这里。', 'Fix a misheard word in the text box before sending, and the fixed word shows up here.')}</span>`, '')])
    + `<div class="grp-l">${L('建议加入 · 从最近的润色差异里找出', 'Suggested · found in recent refinements')}</div>` + grp(suggest)
    + `<div class="grp-l">${L('最近的手动修改', 'Recent hand corrections')}</div>` + grp(fixes);
}

function audio() {
  const inputs = [['', L('系统默认', 'System default')], ...S.inputs.map(n => [n, n])];
  if (S.input && !S.inputs.includes(S.input)) inputs.push([S.input, L(`${S.input}（未连接）`, `${S.input} (not connected)`)]);
  return head(paneTitle('audio'))
    + grp([row(L('输入设备', 'Input device'), select('input', S.input, inputs, L('输入设备', 'Input device')),
      L('每次录音都会重新读取设备；选中的设备不在时用系统默认。', 'Read again for every recording; when the chosen device is missing, the system default is used.'))])
    + grp([
      row(L('录音时静音其他声音', 'Mute other sound while recording'), sw('duck', S.duck, L('录音时静音其他声音', 'Mute other sound while recording')),
        S.duckUnsupported
          ? L('现在的音箱不让系统调音量，所以没法静音它；正在放的音乐仍会暂停。', 'The current speaker has no volume the system can lower, so it cannot be muted; playing music is still paused.')
          : L('使用带回声消除的耳机时自动跳过。录音结束或意外退出后都会恢复。', 'Skipped with an echo-cancelling headset. Sound comes back when recording ends, even after a crash.')),
      row(L('静音时在胶囊里提示', 'Show when muted'), sw('show_ducked', S.prefs.show_ducked, L('静音时在胶囊里提示', 'Show when muted')),
        L('显示一个小喇叭斜杠，让你知道音乐停了是谁干的。', 'A small crossed-out speaker, so you know why the music stopped.')),
    ]);
}

const money = v => v == null ? '—' : v === 0 ? '$0' : v < 0.01 ? '<$0.01' : `$${v.toFixed(2)}`;
const about = v => L(`约 ${money(v)}`, `about ${money(v)}`);
const kilo = n => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
const dayLabel = iso => {
  const [y, m, d] = iso.split('-').map(Number);
  return L(`${m}月${d}日`, new Date(y, m - 1, d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
};

function usage() {
  const u = S.usage;
  if (!u) return head(paneTitle('usage')) + grp([row(`<span class="empty">${L('用量从听写历史里统计。打开「历史与隐私 › 保存听写历史」后开始记录。', 'Usage is counted from the dictation history. It starts once History & Privacy › Save dictation history is on.')}</span>`, '')]);
  const hasCost = u.daily.some(d => d.cost);
  const by = UI.usageBy || (hasCost ? 'cost' : 'n');
  const tile = (label, t) => `<div class="ustat"><small>${label}</small><b>${times(t.n)}</b><span>${t.refined > t.untracked ? about(t.cost) : '—'}</span></div>`;
  const val = d => by === 'cost' ? (d.cost || 0) : d.n;
  const max = Math.max(...u.daily.map(val), by === 'cost' ? 0.01 : 1);
  const bars = u.daily.map((d, i) => {
    const h = val(d) ? Math.max(2, Math.round(val(d) / max * 100)) : 0;
    const tip = `${dayLabel(d.day)} · ${times(d.n)}${d.refined > d.untracked ? ` · ${about(d.cost)}` : ''}`;
    return `<i tabindex="-1" data-tip="${esc(tip)}" aria-label="${esc(tip)}"${i === u.daily.length - 1 ? ' class="now"' : ''}><s style="height:${h}%"></s></i>`;
  }).join('');
  const peak = L('最高 ', 'Peak ') + (by === 'cost' ? money(max) : times(max));
  const m = u.month;
  const cacheRate = m.prompt ? Math.round(m.cached / m.prompt * 100) : null;
  // Every model's refinements as one row: the model is not named in the UI.
  const sum = k => u.models.reduce((a, x) => a + (x[k] || 0), 0);
  const all = { refined: sum('refined'), prompt: sum('prompt'), cached: sum('cached'), completion: sum('completion'),
    cost: u.models.some(x => x.cost == null) && !sum('cost') ? null : sum('cost') };
  const models = all.refined ? [all].map(x => row(L('润色', 'Refinement'),
    `<span class="val">${L(`${x.refined} 次`, `${x.refined} refinement${x.refined === 1 ? '' : 's'}`)} · ${x.cost == null ? L('没有价格', 'no price') : about(x.cost)}${S.trial && S.trial.on ? L('，免费试用承担', ', covered by the free trial') : ''}</span>`,
    x.prompt ? L(`输入 ${kilo(x.prompt)} token（缓存 ${kilo(x.cached)}）· 输出 ${kilo(x.completion)}`, `In ${kilo(x.prompt)} tokens (${kilo(x.cached)} cached) · out ${kilo(x.completion)}`)
      : L('这段时间没有记下 token 数', 'No token counts recorded for this period')))
    : [row(`<span class="empty">${L('最近 30 天没有润色。', 'No refinements in the last 30 days.')}</span>`, '')];
  const what = by === 'cost' ? L('花费', 'spend') : L('听写次数', 'dictations');
  const lastDays = L(`最近 ${u.days} 天`, `Last ${u.days} days`);
  const note = L(
    `只算这台 Mac 上${S.name}发出的润色请求，按 API 返回的 token 数和官方价格估算，语音识别在本地不花钱。${cacheRate != null ? `输入里 ${cacheRate}% 走了缓存，按一成计价。` : ''}${m.untracked ? `有 ${m.untracked} 次是更新前的记录，没有 token 数，只计次数。` : ''}`,
    `Only the refinement requests ${S.name} sent from this Mac, estimated from the token counts the API returned and list prices. Speech recognition runs locally and costs nothing.${cacheRate != null ? ` ${cacheRate}% of the input was cached, billed at a tenth.` : ''}${m.untracked ? ` ${m.untracked} dictations are from before this was recorded and are only counted.` : ''}`);
  return head(paneTitle('usage'))
    + `<div class="ustats">${tile(L('今天', 'Today'), u.today)}${tile(L('最近 7 天', 'Last 7 days'), u.week)}${tile(lastDays, m)}</div>`
    + `<div class="grp uchart"><div class="uhead"><b>${L(`每天${what}`, by === 'cost' ? 'Daily spend' : 'Daily dictations')}</b>${seg('usageBy', by, [['cost', L('花费', 'Spend')], ['n', L('次数', 'Count')]])}</div>`
    + `<div class="ubars" role="img" aria-label="${esc(L(`最近 ${u.days} 天每天的${what}`, `Daily ${what}, last ${u.days} days`))}">${bars}</div>`
    + `<div class="uaxis"><span>${dayLabel(u.daily[0].day)}</span><span class="utip" data-rest="${esc(peak)}">${esc(peak)}</span><span>${L('今天', 'Today')}</span></div></div>`
    + `<div class="grp-l">${L('润色', 'Refinement')} · ${lastDays}</div>` + grp(models)
    + `<p class="note">${esc(note)}</p>`;
}

function privacy() {
  const p = S.prefs, c = UI.confirm;
  const confirmBox = !c ? '' : c.kind === 'days'
    ? `<div class="confirm"><span>${c.n ? L(`会删除 ${c.n} 条 ${esc(DAYS(c.days))}以前的听写。`, `This deletes ${c.n} dictation${c.n === 1 ? '' : 's'} older than ${esc(DAYS(c.days))}.`) : L(`以后只保留最近 ${esc(DAYS(c.days))}。`, `From now on only the last ${esc(DAYS(c.days))} are kept.`)}</span><button class="mbtn" data-act="cancelconfirm">${L('取消', 'Cancel')}</button><button class="mbtn pri" data-act="confirmdays">${c.n ? L('删除并保留', 'Delete and Keep') : L('好', 'OK')}</button></div>`
    : `<div class="confirm"><span>${c.n ? L(`删除全部 ${c.n} 条听写历史？不能撤销。`, `Delete all ${c.n} dictations from the history? This can't be undone.`) : L('现在没有听写历史。', 'There is no dictation history.')}</span><button class="mbtn" data-act="cancelconfirm">${L('取消', 'Cancel')}</button>${c.n ? `<button class="mbtn pri" data-act="confirmclear">${L('全部删除', 'Delete All')}</button>` : ''}</div>`;
  const days = select('history_days', c && c.kind === 'days' ? c.days : p.history_days, S.choices.history_days.map(v => [v, DAYS(v)]), L('保留', 'Keep'));
  const count = S.history.count;
  return head(paneTitle('privacy'))
    + grp([
      row(L('保存听写历史', 'Save dictation history'), sw('save_history', p.save_history, L('保存听写历史', 'Save dictation history')),
        L(`文字、耗时和 App 名称存在这台 Mac 上；${count ? `现在有 ${count} 条，` : ''}从不保存音频。`,
          `Text, timings and app names stay on this Mac${count ? ` (${count} now)` : ''}; audio is never saved.`)),
      row(L('保留', 'Keep'), days),
      row(L('记下发送前的修改', 'Keep edits made before sending'), sw('save_sent_text', p.save_sent_text, L('记下发送前的修改', 'Keep edits made before sending'), !p.save_history),
        L('插入后你手动改过再发送的，会把最后发出去的文字存在这条历史旁边，方便对照。只看刚插入的那个输入框，文档类的大段内容不记。',
          'If you edit the text after it is inserted and then send it, what you sent is kept next to it in the history. Only the field it went into is watched, and long documents are skipped.')),
    ], c && c.kind === 'days' ? confirmBox : '')
    + `<div class="grp-l">${L('发送给润色模型的内容', 'What the refinement model receives')}</div>`
    + grp([
      row(L('原始转写文字', 'The raw transcript'), `<span class="st-ok">${L('发送', 'Sent')}</span>`),
      row(L('当前 App 名称和窗口标题', 'App name and window title'), sw('send_window_title', p.send_window_title, L('发送窗口标题', 'Send the window title')),
        L('用来判断语气。窗口标题可能包含文件名或邮件主题。', 'Used to judge the tone. A window title can contain a file name or an email subject.')),
      row(L('光标前的文字', 'Text before the cursor'), sw('send_before_text', p.send_before_text, L('发送光标前的文字', 'Send the text before the cursor')),
        L('最多约 300 字，用来按上文写对人名、术语和同音字。会多花一点费用和时间，默认关闭；密码框从不读取。',
          'Up to about 300 characters, to get names, terms and homophones right from context. Costs a little more time and money, so it is off by default; password fields are never read.')),
      row(L('所选文字', 'The selected text'), sw('rewrite_selection', p.rewrite_selection, L('发送所选文字', 'Send the selected text')), L('只在“改写所选文字”时发送。', 'Only when rewriting the selection.')),
      row(L('音频', 'Audio'), p.asr_engine === 'qwen'
        ? `<span class="val">${L('发给阿里云识别', 'Sent to Alibaba Cloud')}</span>` : `<span class="val">${L('从不离开这台 Mac', 'Never leaves this Mac')}</span>`,
        p.asr_engine === 'qwen' ? L('语音识别选了千问：录音时音频边说边传到阿里云新加坡，只用来识别。', 'Speech recognition is set to Qwen: audio streams to Alibaba Cloud in Singapore while you talk, only to be recognized.') : ''),
    ])
    + grp([
      row(L('发送匿名使用统计', 'Send anonymous usage stats'), sw('send_usage_stats', p.send_usage_stats, L('发送匿名使用统计', 'Send anonymous usage stats')),
        L('每天一次：当天听写了几次、共多少字、免费试用花了多少，加一个随机编号和版本号。只用来了解产品怎么被使用，从不包含听写内容、App 名称或任何个人信息。',
          'Once a day: how many dictations, how many characters, and what the free trial spent, with a random ID and the version. Only to learn how the app is used; never any dictated text, app names or anything personal.')),
    ])
    + `<div class="grp" style="background:transparent; box-shadow:none">${c && c.kind === 'clear' ? `<div class="grp">${confirmBox}</div>` : ''}<div class="btnrow left"><button class="mbtn" data-act="open" data-what="data">${I('folder')} ${L('在访达中显示数据', 'Show Data in Finder')}</button><button class="mbtn danger" data-act="clear">${I('trash')} ${L('清除历史…', 'Clear History…')}</button></div></div>`;
}

function aboutPane() {
  const a = S.author, f = UI.fb, sending = f.status === 'sending';
  const link = (what, label) => `<button class="mbtn" data-act="open" data-what="${what}">${I('link')} ${esc(label)}</button>`;
  const status = f.status === 'sent' ? `<span class="st-ok">${L('已发送，谢谢！', 'Sent, thank you!')}</span>`
    : f.status === 'failed' ? `<span class="st-err">${esc(f.msg)}</span>` : '';
  return head(paneTitle('about'))
    + `<div class="grp"><div class="abt"><img src="appicon.svg" alt=""><div><b>${esc(S.name)}</b><small>${L('版本', 'Version')} ${esc(S.version || '')}</small></div>`
    + `<button class="mbtn" data-act="open" data-what="releases">${L('检查新版本', 'Check for a new version')}</button></div></div>`
    + `<div class="grp-l">${L('关于作者', 'About the author')}</div>`
    + grp([
      row(`<b>${esc(a.name)}</b>`, ''),
      `<div class="row"><div class="rl">${L('言字是开源项目。如果你有兴趣一起把它做得更好，欢迎在 GitHub 上提 Issue 或 Pull Request，也欢迎通过下面的反馈框联系我。', `${esc(S.name)} is open source. If you’d like to help make it better, issues and pull requests on GitHub are welcome, and so is a note through the feedback box below.`)}</div></div>`,
      `<div class="row"><span class="inline">${link('repo', 'GitHub')}${a.github ? link('github', L('作者主页', 'Author profile')) : ''}${a.links.map(([label], i) => link(`link${i}`, label)).join('')}</span></div>`,
    ])
    + `<div class="grp-l">${L('反馈', 'Feedback')}</div>`
    + `<div class="grp fb"><textarea id="fb-text" class="fbtext" maxlength="4000" placeholder="${L('遇到了什么问题，或者想要什么功能？', 'What went wrong, or what would you like to see?')}" aria-label="${L('反馈内容', 'Feedback')}"${sending ? ' disabled' : ''}>${esc(f.text)}</textarea>`
    + `<div class="row"><div class="rl">${L('回复邮箱（可选）', 'Reply email (optional)')}</div><input class="field" id="fb-email" type="email" maxlength="200" value="${esc(f.email)}" aria-label="${L('回复邮箱', 'Reply email')}"${sending ? ' disabled' : ''}></div>`
    + `<div class="row"><label class="rl"><input type="checkbox" id="fb-diag"${f.diag ? ' checked' : ''}${sending ? ' disabled' : ''}> ${L('附上诊断信息', 'Attach diagnostics')}<small>${L('设置概要和最近的日志（已去掉密钥）。日志可能引用少量识别到的词，默认不附。', 'A settings summary and the latest log lines, with keys removed. The log can quote a few recognised words, so it is off by default.')}</small></label></div>`
    + `<div class="btnrow">${status}<button class="mbtn pri" id="fb-send" data-act="feedback"${sending || !f.text.trim() ? ' disabled' : ''}>${sending ? '<span class="spin"></span>' : L('发送', 'Send')}</button></div></div>`
    + `<p class="note">${L('随反馈自动发送：App 版本、macOS 版本、Mac 型号、听写语言设置和界面语言。从不包含听写内容。', 'Sent with it: the app version, macOS version, Mac model, the spoken-language setting and the interface language. Never any dictation text.')}</p>`;
}

const RENDER = { general, dictation, keys, model, vocab, audio, usage, privacy, about: aboutPane };

/* ---------------- drawing ---------------- */
function drawSide() {
  $('#side').innerHTML = PANES.map(([id, t, ic, c]) =>
    `<button role="tab" aria-selected="${id === pane}" data-pane="${id}"><span class="tile" style="background:${c}">${I(ic)}</span>${esc(L(...t))}</button>`).join('')
    + `<div class="foot">${S ? esc(`${S.name} ${S.version || ''}`) : ''}</div>`;
}

function draw() {
  if (!S) return;
  // Keep what is being typed across a redraw.
  const a = document.activeElement, id = a && a.id, val = a && 'value' in a ? a.value : null;
  const sel = a && a.selectionStart != null ? [a.selectionStart, a.selectionEnd] : null;
  drawSide();
  const main = $('#panes'), top = main.scrollTop;
  main.innerHTML = `<section class="pane" role="tabpanel" aria-label="${esc(paneTitle(pane))}">${RENDER[pane]()}</section>`;
  main.scrollTop = top;
  if (id) {
    const el = document.getElementById(id);
    if (el) { if (val != null) el.value = val; el.focus(); if (sel) try { el.setSelectionRange(sel[0], sel[1]); } catch (e) { /* select */ } }
  }
  reportGlass();
}

function show(id) {
  if (!RENDER[id]) return;
  pane = id;
  UI.confirm = null;
  draw();
  $('#panes').scrollTop = 0;
}

/* ---------------- native glass under the sidebar ---------------- */
let lastGeo = '';
function reportGlass() {
  const r = $('#side').getBoundingClientRect();
  const s = [{ id: 'side', x: r.left, y: r.top, w: r.width, h: r.height, r: parseFloat(getComputedStyle($('#side')).borderTopLeftRadius) || 18, a: 1, glass: true, hit: false }];
  const key = JSON.stringify(s);
  if (key !== lastGeo) { lastGeo = key; post({ t: 'geo', s }); }
}
window.addEventListener('resize', reportGlass);

/* ---------------- input ---------------- */
function value(el) {
  if (el.classList.contains('sw')) return el.getAttribute('aria-checked') !== 'true';
  if (el.dataset.v != null) return el.dataset.v;
  return el.value;
}
const NUMERIC = new Set(['dismiss_seconds', 'max_minutes', 'history_days']);

function setting(el) {
  const key = el.dataset.set;
  let v = value(el);
  if (key === 'usageBy') { UI.usageBy = v; draw(); return; }  // only how this page draws
  if (NUMERIC.has(key)) v = Number(v);
  if (key === 'history_days' && v !== 0 && v !== S.prefs.history_days) {
    // Shortening the history deletes what is older: ask first, in place.
    UI.confirm = { kind: 'days', days: v, n: null };
    post({ t: 'count', days: v });
    draw();
    return;
  }
  // Optimistic: switches flip at once; Python's next state settles it.
  if (key in S.prefs) S.prefs[key] = v;
  if (key === 'duck') S.duck = v;
  post({ t: 'set', key, value: v });
  draw();
}

// The usage chart: the axis line under it reads out the day under the pointer.
document.addEventListener('pointerover', e => {
  const tip = document.querySelector('.utip');
  if (!tip) return;
  const bar = e.target.closest('.ubars i');
  tip.textContent = bar ? bar.dataset.tip : tip.dataset.rest;
  tip.classList.toggle('on', !!bar);
});

document.addEventListener('click', e => {
  const tab = e.target.closest('[data-pane]');
  if (tab) { show(tab.dataset.pane); return; }
  const set = e.target.closest('button[data-set]');
  if (set && !set.disabled) { setting(set); return; }
  const act = e.target.closest('[data-act]');
  if (!act || act.disabled) return;
  const a = act.dataset.act;
  if (a === 'open') post({ t: 'open', what: act.dataset.what });
  else if (a === 'test') { UI.test = { busy: true }; post({ t: 'test' }); draw(); }
  else if (a === 'editkey') { UI.keyEdit = act.dataset.env; delete UI.keyMsg[UI.keyEdit]; draw(); const f = document.getElementById(`key-${UI.keyEdit}`); if (f) f.focus(); }
  else if (a === 'cancelkey') { UI.keyEdit = ''; draw(); }
  else if (a === 'savekey') saveKey(act.dataset.env);
  else if (a === 'migrate') post({ t: 'migrate' });
  else if (a === 'word') post({ t: 'vocab', op: 'add', term: act.dataset.term });
  else if (a === 'unword') post({ t: 'vocab', op: 'remove', term: act.dataset.term });
  else if (a === 'unlearn') post({ t: 'vocab', op: 'unlearn', term: act.dataset.term });
  else if (a === 'reject') post({ t: 'vocab', op: 'reject', term: act.dataset.term });
  else if (a === 'feedback') sendFeedback();
  else if (a === 'clear') { UI.confirm = { kind: 'clear', n: S.history.count }; draw(); }
  else if (a === 'cancelconfirm') { UI.confirm = null; draw(); }
  else if (a === 'confirmdays') { const d = UI.confirm.days; UI.confirm = null; S.prefs.history_days = d; post({ t: 'set', key: 'history_days', value: d }); draw(); }
  else if (a === 'confirmclear') { UI.confirm = null; post({ t: 'clear' }); draw(); }
});

document.addEventListener('input', e => {
  if (e.target.id === 'fb-text') {
    UI.fb.text = e.target.value;
    const send = document.getElementById('fb-send');
    if (send) send.disabled = !UI.fb.text.trim();
  } else if (e.target.id === 'fb-email') UI.fb.email = e.target.value;
});

document.addEventListener('change', e => {
  if (e.target.id === 'fb-diag') { UI.fb.diag = e.target.checked; return; }
  const el = e.target.closest('select[data-set]');
  if (el) setting(el);
});

document.addEventListener('keydown', e => {
  if (e.target.id === 'newword' && e.key === 'Enter') {
    const term = e.target.value.trim();
    if (term) { post({ t: 'vocab', op: 'add', term }); e.target.value = ''; }
    e.preventDefault();
  } else if (e.target.id && e.target.id.startsWith('key-')) {
    if (e.key === 'Enter') { e.preventDefault(); saveKey(e.target.id.slice(4)); }
    if (e.key === 'Escape') { e.preventDefault(); UI.keyEdit = ''; draw(); }
  } else if (e.key === 'Escape' && UI.confirm) { UI.confirm = null; draw(); }
  else if ((e.metaKey || e.ctrlKey) && /^[1-9]$/.test(e.key)) { e.preventDefault(); show(PANES[+e.key - 1][0]); }
});

function sendFeedback() {
  const f = UI.fb;
  if (!f.text.trim()) return;
  if (f.email.trim() && !/^\S+@\S+\.\S+$/.test(f.email.trim())) {
    f.status = 'failed'; f.msg = L('回复邮箱看起来不对，改一下或清空再发。', 'That reply email does not look right; fix it or clear it.'); draw(); return;
  }
  f.status = 'sending';
  post({ t: 'feedback', message: f.text, email: f.email.trim(), diagnostics: f.diag });
  draw();
}

function saveKey(envName) {
  const f = document.getElementById(`key-${envName}`);
  const v = f ? f.value.trim() : '';
  if (!v) { if (f) f.focus(); return; }
  UI.keyBusy = envName;
  post({ t: 'key', env: envName, value: v });
  draw();
}

/* ---------------- from Python ---------------- */
on('env', m => {
  env(m);
  document.body.classList.toggle('has-glass', !!m.native);
  document.title = L('设置', 'Settings');
  $('#side').setAttribute('aria-label', L('设置分区', 'Settings sections'));
  if (S) draw(); else drawSide();
});
on('state', m => { S = m; draw(); });
on('pane', m => show(m.id));
on('keyResult', m => {
  UI.keyBusy = '';
  UI.keyMsg[m.env] = { ok: m.ok, text: m.msg };
  if (m.ok) UI.keyEdit = '';
  draw();
});
on('feedbackResult', m => {
  const f = UI.fb;
  f.status = m.ok ? 'sent' : 'failed';
  f.msg = m.msg || '';
  if (m.ok) f.text = '';  // kept on failure so nothing is lost
  draw();
});
on('testResult', m => { UI.test = m; draw(); });
on('count', m => { if (UI.confirm && UI.confirm.kind === 'days' && UI.confirm.days === m.days) { UI.confirm.n = m.n; draw(); } });
on('vocabResult', m => { UI.vocabMsg = m.msg || ''; draw(); });

drawSide();
post({ t: 'ready' });
})();
