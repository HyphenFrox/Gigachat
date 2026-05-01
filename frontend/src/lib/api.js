/**
 * Thin REST client for the FastAPI backend.
 * All paths are relative so they flow through the Vite proxy (in dev) or
 * are served from the same origin (in production).
 *
 * Every function throws on non-2xx responses — callers handle errors by
 * showing a Sonner toast near the call site.
 */

/**
 * Low-level fetch wrapper. Normalizes errors so the throw has a useful message.
 *
 * @param {string} path
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
async function request(path, options = {}) {
  // Default JSON content type only when a body is being sent. For multipart
  // uploads (FormData) the browser must set the boundary, so we explicitly
  // skip Content-Type in that case.
  const isFormData =
    typeof FormData !== 'undefined' && options.body instanceof FormData
  const headers = {
    ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
    ...(options.headers || {}),
  }
  const res = await fetch(path, { ...options, headers })
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = body.detail || body.error || JSON.stringify(body)
    } catch {
      detail = await res.text().catch(() => '')
    }
    // Late 401: our session cookie expired (30-day TTL) or the admin
    // rotated the secret. Notify the top-level app so it can flip back
    // to the login screen instead of the surface-level toast the caller
    // would otherwise show. Login/status endpoints are exempted — they
    // legitimately 401 on a bad password and the caller handles that.
    if (
      res.status === 401 &&
      typeof window !== 'undefined' &&
      !path.startsWith('/api/auth/')
    ) {
      window.dispatchEvent(new CustomEvent('gigachat:unauthorized'))
    }
    throw new Error(`${res.status} ${res.statusText}${detail ? ` — ${detail}` : ''}`)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  /**
   * Auth status check — returns `{requires_password, authenticated, host}`.
   * When `requires_password` is false the server is bound to loopback and
   * the login flow is skipped entirely. Callers use the `authenticated`
   * flag to decide whether to render LoginView or the main app.
   */
  getAuthStatus: () => request('/api/auth/status'),

  /**
   * Exchange a typed password for a session cookie. The server sets an
   * httponly cookie so subsequent requests authenticate automatically —
   * the caller doesn't need to hold onto the returned token field.
   */
  login: (password) =>
    request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ password }),
    }),

  /** Clear the session cookie. */
  logout: () => request('/api/auth/logout', { method: 'POST' }),

  /**
   * List installed Ollama models. By default the backend filters the result to
   * models whose `capabilities` include `tools` — the agent loop 400s on
   * anything else. Pass `{ all: true }` to bypass that filter (shows every
   * installed model, including embedding-only or non-tool chat models).
   */
  listModels: ({ all = false } = {}) =>
    request(`/api/models${all ? '?all=1' : ''}`),

  /**
   * Fetch the capabilities list for a single installed model. Used by the
   * composer to warn the user before attaching images to a non-vision
   * chat model (the attachment would be stripped by the agent layer).
   *
   * Response: {model, capabilities: [...], vision: boolean, tools: boolean}
   */
  getModelCapabilities: (name) =>
    request(`/api/models/${encodeURIComponent(name)}/capabilities`),

  /**
   * Fetch the auto-detected host profile (RAM, VRAM, GPU name) plus the
   * context-window size the backend actually picked. The frontend uses
   * `num_ctx` as the denominator in its token-usage gauge so the UI never
   * drifts out of sync with what Ollama is running.
   */
  getSystemConfig: () => request('/api/system/config'),

  /** List all stored conversations, newest first (pinned float to top). */
  listConversations: () => request('/api/conversations'),

  /**
   * Substring-search conversations by title, tags, or message content.
   * Empty/whitespace query returns no results — caller should fall back to
   * `listConversations` for the unfiltered view.
   */
  searchConversations: (q) =>
    request(`/api/conversations/search?q=${encodeURIComponent(q || '')}`),

  /**
   * Cross-conversation semantic search. Embeds the query locally and returns
   * the top-N most similar messages (meaning-based, not keyword).
   *
   * Response: {hits: [{message_id, conversation_id, conversation_title,
   *                    role, snippet, score, created_at}],
   *            indexed, total, error?}
   * `error === "embedding_unavailable"` signals the nomic-embed-text Ollama
   * model isn't installed or Ollama is offline — caller should toast the user
   * with a hint to run `ollama pull nomic-embed-text`.
   */
  semanticSearchConversations: (q) =>
    request(
      `/api/conversations/semantic-search?q=${encodeURIComponent(q || '')}`,
    ),

  /**
   * One-shot backfill for messages created before semantic search shipped.
   * Processes up to 500 un-embedded messages per call; call again if more
   * are pending.
   */
  reindexEmbeddings: () =>
    request('/api/conversations/reindex', { method: 'POST' }),

  /**
   * Create a new conversation.
   * @param {{title?:string, model?:string, cwd?:string, auto_approve?:boolean, permission_mode?:string}} body
   */
  createConversation: (body) =>
    request('/api/conversations', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Fetch one conversation's metadata and its full message history. */
  /**
   * Fetch one conversation's metadata and a page of its messages.
   *
   * - No options → entire transcript (legacy behaviour, backwards-compatible).
   * - `{ limit }` → most-recent N messages plus a `total` count + `has_more`
   *   flag. Used as the initial load for chat-app-style scroll-up paging.
   * - `{ limit, beforeId }` → the N messages strictly older than
   *   `beforeId`. Used by the scroll-up "load more" gesture.
   */
  getConversation: (id, opts = null) => {
    if (!opts) return request(`/api/conversations/${id}`)
    const qs = new URLSearchParams()
    if (opts.limit) qs.set('limit', String(opts.limit))
    if (opts.beforeId) qs.set('before_id', opts.beforeId)
    const sep = qs.toString() ? `?${qs.toString()}` : ''
    return request(`/api/conversations/${id}${sep}`)
  },

  /** Patch title / model / cwd / permission_mode / quality_mode / pinned / tags / persona / budget_* on a conversation. */
  updateConversation: (id, patch) =>
    request(`/api/conversations/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /**
   * Append a follow-up user message to the *currently running* turn's input
   * queue. Returns immediately — the existing SSE stream will emit a
   * `user_message_added` event once the agent loop drains the queue.
   *
   * @param {string} convId
   * @param {{content:string, images?:string[]}} body
   */
  queueMessage: (convId, body) =>
    request(`/api/conversations/${convId}/queue`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Delete a conversation (messages cascade). */
  deleteConversation: (id) =>
    request(`/api/conversations/${id}`, { method: 'DELETE' }),

  /** Approve or reject a pending tool call the agent is waiting on. */
  approve: (id, approvalId, approved) =>
    request(`/api/conversations/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ approval_id: approvalId, approved }),
    }),

  /**
   * Resolve a pending AskUserQuestion — the agent is waiting on a multi-choice
   * button click. `value` must match one of the option values the agent emitted.
   */
  answerQuestion: (id, answerId, value) =>
    request(`/api/conversations/${id}/answer`, {
      method: 'POST',
      body: JSON.stringify({ answer_id: answerId, value }),
    }),

  /**
   * Switch a plan-mode conversation out of plan mode and enqueue the most
   * recent assistant plan as the next user turn. Server-side flips
   * permission_mode → approve_edits and inserts into queued_inputs.
   */
  executePlan: (id) =>
    request(`/api/conversations/${id}/execute-plan`, { method: 'POST' }),

  /* -------------------- Side tasks (spawn_task chips) ------------------- */

  /** List pending side-task chips for this conversation. */
  listSideTasks: (id) =>
    request(`/api/conversations/${id}/side-tasks`),

  /**
   * Open a side-task chip — spins a new conversation with the stored prompt.
   * Returns `{ ok, conversation }` so the caller can both navigate to the new
   * chat and insert its row into the sidebar without a second round-trip.
   */
  openSideTask: (sid, overrides = {}) =>
    request(`/api/side-tasks/${sid}/open`, {
      method: 'POST',
      body: JSON.stringify(overrides),
    }),

  /** Dismiss a side-task chip (transition to terminal 'dismissed' state). */
  dismissSideTask: (sid) =>
    request(`/api/side-tasks/${sid}/dismiss`, { method: 'POST' }),

  /* -------------------- Git worktrees (isolation) ----------------------- */

  /** List worktrees (active + removed) the agent created in this conversation. */
  listWorktrees: (id) =>
    request(`/api/conversations/${id}/worktrees`),

  /**
   * Ask the server to stop the currently-running agent turn for this
   * conversation. The server flips a flag that the agent loop polls at
   * well-defined checkpoints — so stopping is felt even if the local
   * model is mid-generation and wouldn't otherwise notice the SSE
   * connection closing.
   */
  stopTurn: (id) =>
    request(`/api/conversations/${id}/stop`, { method: 'POST' }),

  /**
   * Upload a pasted/dropped image OR document. The server branches on the
   * file's Content-Type: images are stored verbatim for multimodal input,
   * documents (pdf/txt/md/csv) have their text extracted server-side.
   *
   * Response shape:
   *   {
   *     name, size, content_type, original_name, kind: 'image'|'document',
   *     extracted_text?, truncated?, page_count?, extract_error?
   *   }
   *
   * For images the caller passes `name` in the next message's `images` array.
   * For documents the caller prepends `extracted_text` (with a small header
   * identifying the file) to the message body so the model sees the content.
   *
   * @param {string} convId
   * @param {File|Blob} file
   */
  uploadAttachment: (convId, file) => {
    const fd = new FormData()
    // Give anonymous blobs a filename the server can classify by extension.
    const defaultName = `paste.${(file.type || 'image/png').split('/')[1] || 'png'}`
    const filename = file.name || defaultName
    fd.append('file', file, filename)
    return request(`/api/conversations/${convId}/uploads`, {
      method: 'POST',
      body: fd,
    })
  },

  /** Restore files from a checkpoint stamp. */
  restoreCheckpoint: (id, stamp) =>
    request(`/api/conversations/${id}/restore/${stamp}`, { method: 'POST' }),

  /**
   * Pin or unpin a single message. Pinned messages are exempt from
   * auto-compaction — use this for "don't let the model forget this".
   */
  pinMessage: (convId, messageId, pinned) =>
    request(`/api/conversations/${convId}/messages/${messageId}`, {
      method: 'PATCH',
      body: JSON.stringify({ pinned: !!pinned }),
    }),

  /** Permanently delete one message from a conversation. */
  deleteMessage: (convId, messageId) =>
    request(`/api/conversations/${convId}/messages/${messageId}`, {
      method: 'DELETE',
    }),

  /** List every pinned message in one conversation (oldest-first). */
  listPinnedMessages: (convId) =>
    request(`/api/conversations/${convId}/pinned`),

  /** Read the per-conversation memory markdown file (empty string if none). */
  getConversationMemory: (convId) =>
    request(`/api/conversations/${convId}/memory`),

  /** Overwrite the per-conversation memory markdown file. */
  putConversationMemory: (convId, content) =>
    request(`/api/conversations/${convId}/memory`, {
      method: 'PUT',
      body: JSON.stringify({ content: content || '' }),
    }),

  /** Clear the per-conversation memory file entirely. */
  deleteConversationMemory: (convId) =>
    request(`/api/conversations/${convId}/memory`, { method: 'DELETE' }),

  /** Cumulative usage + budget snapshot (drives the budget gauge). */
  getConversationUsage: (convId) =>
    request(`/api/conversations/${convId}/usage`),

  /* -------------------- Lifecycle hooks -------------------------------- */
  /** List every hook (enabled + disabled) plus the set of valid event names. */
  listHooks: () => request('/api/hooks'),

  /**
   * Register a new hook. `event` must be one of the names returned by
   * listHooks(). The body of the command is a shell string run under /bin/sh
   * (or cmd on Windows via shell=True) with a JSON payload on stdin.
   */
  createHook: (body) =>
    request('/api/hooks', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Patch fields (event/command/matcher/timeout_seconds/enabled) on a hook. */
  updateHook: (id, patch) =>
    request(`/api/hooks/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Remove a hook permanently. Use enabled=false if you only want to pause it. */
  deleteHook: (id) => request(`/api/hooks/${id}`, { method: 'DELETE' }),

  /* -------------------- Global memories -------------------------------- */
  /**
   * List every cross-conversation "global" memory note. These are durable
   * facts the user (or agent, via remember(scope='global')) has saved that
   * get injected into every conversation's system prompt.
   */
  listMemories: () => request('/api/memories'),

  /**
   * Add a new global memory.
   * @param {{content:string, topic?:string|null}} body
   */
  createMemory: (body) =>
    request('/api/memories', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * Patch content/topic on an existing memory in place.
   * @param {string} id
   * @param {{content?:string, topic?:string|null}} patch
   */
  updateMemory: (id, patch) =>
    request(`/api/memories/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Permanently delete a global memory entry. */
  deleteMemory: (id) => request(`/api/memories/${id}`, { method: 'DELETE' }),

  /* -------------------- Secrets (API tokens / creds) ------------------- */
  /**
   * List every stored secret's *metadata* — the response never includes the
   * raw value. Call `revealSecret(id)` to fetch the plaintext for a single
   * entry when the user clicks "reveal" in the UI.
   */
  listSecrets: () => request('/api/secrets'),

  /** Fetch one secret including its plaintext value. */
  revealSecret: (id) => request(`/api/secrets/${id}`),

  /**
   * Store a new secret. Name must be [A-Za-z_][A-Za-z0-9_]{0,63} so it's
   * safe to reference as `{{secret:NAME}}` from tool calls.
   * @param {{name:string, value:string, description?:string|null}} body
   */
  createSecret: (body) =>
    request('/api/secrets', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * Patch a secret in place. Omit a field to leave it unchanged.
   * @param {string} id
   * @param {{name?:string, value?:string, description?:string|null}} patch
   */
  updateSecret: (id, patch) =>
    request(`/api/secrets/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Permanently delete a stored secret. */
  deleteSecret: (id) => request(`/api/secrets/${id}`, { method: 'DELETE' }),

  /* -------------------- Compute pool (multi-PC workers) ----------------- */
  /**
   * List every registered compute worker.
   * Auth tokens are NEVER returned — each row carries `auth_token_set: bool`.
   * Response: `{workers: [...]}`.
   */
  listComputeWorkers: () => request('/api/compute-workers'),

  /**
   * Register a new worker.
   *
   * `address` is a LAN hostname (`worker.local`) or a private IPv4
   * (`192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`); the probe layer strips
   * any `http(s)://` prefix the user pastes. All ongoing traffic flows
   * over this address.
   *
   * `tailscale_host` is optional — a stable Tailscale identifier (MagicDNS
   * name like `worker.your-tailnet.ts.net`, or a CGNAT IPv4 in
   * 100.64.0.0/10). It's used ONLY by the auto-repair routine: when the
   * worker reconnects to the LAN with a different DHCP lease the backend
   * reaches it over Tailscale to rediscover the new LAN IP, then resumes
   * regular traffic over LAN.
   *
   * @param {{label:string,address:string,ollama_port?:number,
   *          auth_token?:string|null,ssh_host?:string|null,
   *          tailscale_host?:string|null,enabled?:boolean,
   *          use_for_chat?:boolean,use_for_embeddings?:boolean,
   *          use_for_subagents?:boolean}} body
   */
  createComputeWorker: (body) =>
    request('/api/compute-workers', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * Patch a worker in place. Send `auth_token: ""` to clear the token,
   * `null`/omit to leave it alone, or a new string to replace it.
   * @param {string} id
   * @param {object} patch
   */
  updateComputeWorker: (id, patch) =>
    request(`/api/compute-workers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Permanently delete a worker row. */
  deleteComputeWorker: (id) =>
    request(`/api/compute-workers/${id}`, { method: 'DELETE' }),

  /**
   * Manually probe one worker now. Backend pings its Ollama and returns
   * `{ok, capabilities:{version, models[]}, error, last_seen}`. Capabilities
   * are also persisted on the row, so a follow-up `listComputeWorkers` will
   * show the same data.
   */
  probeComputeWorker: (id) =>
    request(`/api/compute-workers/${id}/probe`, { method: 'POST' }),

  /** Probe every enabled worker now. Returns `{results: [{worker_id, label, ok, error}]}`. */
  probeAllComputeWorkers: () =>
    request('/api/compute-workers/probe-all', { method: 'POST' }),

  /**
   * LAN-first model copy: scp this host's Ollama-managed model blobs to
   * the named worker. Requires the worker row to have `ssh_host` set.
   * Returns a summary `{ok, blobs_total, blobs_already_present, blobs_shipped, bytes_shipped}`.
   */
  pushModelToWorker: (workerId, modelName) =>
    request(
      `/api/compute-workers/${workerId}/push-model?model=${encodeURIComponent(modelName)}`,
      { method: 'POST' },
    ),

  /** Preview what `pushModelToWorker` would ship. */
  pushModelPlan: (workerId, modelName) =>
    request(
      `/api/compute-workers/${workerId}/push-model/plan?model=${encodeURIComponent(modelName)}`,
    ),

  /**
   * Where on the LAN is `model` already available? Returns
   * `{source: {kind: 'host'|'worker', worker_id?, label?} | null}`.
   */
  findLanSource: (modelName, excludeWorkerId = null) => {
    const q = new URLSearchParams({ model: modelName })
    if (excludeWorkerId) q.set('exclude_worker_id', excludeWorkerId)
    return request(`/api/models/lan-source?${q}`)
  },

  /* -------------------- Split models (Phase 2) -------------------------- */
  /**
   * List every registered split model + the host's llama.cpp install
   * status. Response: `{split_models: [...], llama_cpp: {...}}`.
   */
  listSplitModels: () => request('/api/split-models'),

  /**
   * Register a new split-model definition (just the row — doesn't
   * spawn llama-server yet). Call `startSplitModel(id)` after to bring
   * the server up.
   * @param {{label:string, gguf_path:string, worker_ids?:string[],
   *          llama_port?:number, enabled?:boolean}} body
   */
  createSplitModel: (body) =>
    request('/api/split-models', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Patch a split-model row's user-facing fields. */
  updateSplitModel: (id, patch) =>
    request(`/api/split-models/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Stop and remove a split-model row. */
  deleteSplitModel: (id) =>
    request(`/api/split-models/${id}`, { method: 'DELETE' }),

  /** Spawn llama-server for this row; resolves once /health says OK. */
  startSplitModel: (id) =>
    request(`/api/split-models/${id}/start`, { method: 'POST' }),

  /** Terminate the llama-server child. Idempotent. */
  stopSplitModel: (id) =>
    request(`/api/split-models/${id}/stop`, { method: 'POST' }),

  /** Read-only status snapshot — the UI polls this while a row is `loading`. */
  getSplitModelStatus: (id) =>
    request(`/api/split-models/${id}/status`),

  /**
   * Trigger an explicit llama.cpp download + install on the host.
   * Synchronous (multi-hundred-MB download — the UI shows a spinner).
   * Variant defaults to `host` (CUDA build for the GPU).
   */
  installLlamaCpp: (variant = 'host') =>
    request(
      `/api/split-models/install-llamacpp?variant=${encodeURIComponent(variant)}`,
      { method: 'POST' },
    ),

  /* -------------------- Web Push notifications -------------------------- */
  /**
   * Count of browsers currently registered for push. Used by the settings UI
   * to render "Enabled on N devices" without exposing endpoint URLs.
   */
  pushStatus: () => request('/api/push/status'),

  /* -------------------- MCP (Model Context Protocol) ------------------- */
  /**
   * List every configured MCP server with its live status.
   * Response: {servers: [{id, name, command, args, env, enabled,
   *            status: {running, tools[], stderr_tail?, error?}}]}
   */
  listMcpServers: () => request('/api/mcp/servers'),

  /**
   * Add a new MCP server configuration. The backend tries to bring it up
   * immediately and returns both the persisted row and a per-server
   * refresh report the UI can show as a success/error toast.
   * @param {{name:string, command:string, args?:string[], env?:object, enabled?:boolean}} body
   */
  createMcpServer: (body) =>
    request('/api/mcp/servers', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * Patch any subset of name/command/args/env/enabled on an existing row.
   * A change to command/args/env forces the subprocess to restart.
   */
  updateMcpServer: (id, patch) =>
    request(`/api/mcp/servers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Permanently remove an MCP server config and terminate its subprocess. */
  deleteMcpServer: (id) =>
    request(`/api/mcp/servers/${id}`, { method: 'DELETE' }),

  /**
   * Force a reconcile of live sessions against the DB rows — used after the
   * user fixes a misconfigured server and wants to retry without a full
   * backend restart.
   */
  refreshMcpServers: () =>
    request('/api/mcp/refresh', { method: 'POST' }),

  /* -------------------- User settings ---------------------------------- */
  /**
   * Fetch every stored user setting plus the currently-effective default
   * chat model. The settings panel uses this to render a picker that defaults
   * to the user's saved choice, and falls back to a "(auto-detected)" label
   * when nothing is set.
   *
   * Response: {settings: {default_chat_model: string|null}, effective_chat_model: string}
   */
  getSettings: () => request('/api/settings'),

  /**
   * Patch one or more user settings. Sending `default_chat_model: null` (or
   * an empty string) clears the user's override so the auto-tune
   * recommendation takes over again.
   *
   * @param {{default_chat_model?: string|null}} patch
   */
  updateSettings: (patch) =>
    request('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /* -------------------- Scheduled tasks -------------------------------- */
  /**
   * List every pending scheduled task, soonest-first.
   * Response: {tasks: [{id, name, prompt, next_run_at, interval_seconds, cwd, created_at}]}
   */
  listScheduledTasks: () => request('/api/scheduled-tasks'),

  /**
   * Create a new scheduled task. `run_at` accepts unix-seconds or an
   * ISO-8601 string (the <input type="datetime-local"> value is fine).
   * Pass `interval_seconds` > 0 for a recurring job, or omit / null for a
   * one-shot. Backend enforces a 60s minimum interval.
   *
   * @param {{name:string, prompt:string, run_at:string|number,
   *          interval_seconds?:number|null, cwd?:string|null}} body
   */
  createScheduledTask: (body) =>
    request('/api/scheduled-tasks', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Cancel a pending task. Accepts full ids as well as short-id prefixes. */
  deleteScheduledTask: (tid) =>
    request(`/api/scheduled-tasks/${tid}`, { method: 'DELETE' }),

  /**
   * Get the active autonomous-loop row for a conversation (or null). The
   * ChatView polls this on mount + after every turn so the banner can
   * appear/disappear without a dedicated SSE event.
   * Response: {loop: {id, prompt, interval_seconds, next_run_at, ...}|null}
   */
  getConversationLoop: (cid) =>
    request(`/api/conversations/${cid}/loop`),

  /**
   * Stop the autonomous loop on a conversation (idempotent — safe even if no
   * loop is running). Mirrors the `stop_loop` tool the agent can call itself.
   */
  stopConversationLoop: (cid) =>
    request(`/api/conversations/${cid}/loop`, { method: 'DELETE' }),

  /**
   * Fetch the codebase-index status for a conversation's cwd.
   * Response: {index: {status, file_count, chunk_count, last_indexed_at,
   *                    error, ...}|null, cwd}
   */
  getCodebaseIndex: (cid) =>
    request(`/api/conversations/${cid}/codebase-index`),

  /**
   * Force a background re-index of this conversation's cwd. Returns the
   * fresh status so the UI can switch to "indexing" instantly.
   */
  reindexCodebase: (cid) =>
    request(`/api/conversations/${cid}/codebase-index/reindex`, {
      method: 'POST',
    }),

  /**
   * Fuzzy-find files in the conversation's cwd for @-mention autocomplete.
   * The backend does a name substring match and, for long-form queries,
   * folds in semantic hits from the codebase index.
   *
   * Response: {files: [{path, rel_path, name, snippet?, source}], cwd, query}
   */
  searchConversationFiles: (cid, q, limit = 12) => {
    const qs = new URLSearchParams({ q: q || '', limit: String(limit) })
    return request(`/api/conversations/${cid}/files/search?${qs.toString()}`)
  },

  /* -------------------- Docs URL indexing ------------------------------ */
  /**
   * List every indexed documentation URL with its crawl status.
   * Response: {urls: [{id, url, title, status, pages_crawled, chunk_count,
   *                     max_pages, same_origin_only, error, last_indexed_at,
   *                     created_at, updated_at}]}
   */
  listDocUrls: () => request('/api/docs/urls'),

  /** Fetch one URL row — used to poll crawl progress. */
  getDocUrl: (did) => request(`/api/docs/urls/${did}`),

  /**
   * Register a new URL and kick off the first crawl.
   * @param {{url:string, title?:string|null, max_pages?:number,
   *          same_origin_only?:boolean}} body
   */
  createDocUrl: (body) =>
    request('/api/docs/urls', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Force a fresh crawl of an existing seed (overwrites cached chunks). */
  reindexDocUrl: (did) =>
    request(`/api/docs/urls/${did}/reindex`, { method: 'POST' }),

  /** Drop the URL and every chunk crawled beneath it. */
  deleteDocUrl: (did) =>
    request(`/api/docs/urls/${did}`, { method: 'DELETE' }),

  /* -------------------- User-defined tools ----------------------------- */
  /**
   * List every user-minted tool (enabled + disabled). These are Python
   * snippets the agent (or the user via the Tools tab) created at runtime —
   * each one is callable from the agent loop as if it were a built-in.
   *
   * Response: {tools: [{id, name, description, schema, code, deps, category,
   *                     timeout_seconds, enabled, created_at, updated_at}],
   *            disabled: boolean}
   * `disabled === true` means GIGACHAT_DISABLE_USER_TOOLS is set — the UI
   * should show a banner and prevent creation.
   */
  listUserTools: () => request('/api/user-tools'),

  /**
   * Create a new user tool. The server validates the Python code with AST,
   * installs any pip deps into a shared sandboxed venv, and persists the
   * tool so it survives restarts.
   *
   * Response: {tool: row, install_log: string} — install_log captures pip's
   * stdout/stderr so a "failed dep" error can be shown to the user verbatim.
   *
   * @param {{name:string, description:string, code:string, schema?:object,
   *          deps?:string[], category?:'read'|'write', timeout_seconds?:number}} body
   */
  createUserTool: (body) =>
    request('/api/user-tools', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * Patch fields on a user tool (name is immutable). Useful for toggling
   * enabled, tweaking the description, or bumping the timeout.
   *
   * @param {string} id
   * @param {{description?:string, code?:string, schema?:object,
   *          deps?:string[], category?:string, timeout_seconds?:number,
   *          enabled?:boolean}} patch
   */
  updateUserTool: (id, patch) =>
    request(`/api/user-tools/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  /** Permanently delete a user tool. Installed deps in the venv are kept. */
  deleteUserTool: (id) =>
    request(`/api/user-tools/${id}`, { method: 'DELETE' }),

  /* -------------------- Filesystem ------------------------------------- */
  /**
   * Open a native folder-picker dialog on the host machine and return the
   * selected path. Response shape: `{ok: true, path: string|null}`.
   * `path === null` means the user cancelled the dialog.
   *
   * The dialog is backed by tkinter on the server side, so it only works
   * when the backend runs locally on the same machine as the UI — which is
   * the intended single-user deployment.
   */
  pickDirectory: () =>
    request('/api/fs/pick-directory', { method: 'POST' }),

  /**
   * Fetch the backend's default working directory (the folder Gigachat
   * itself runs from). Used as the pre-filled value in the new-chat
   * dialog so a user who doesn't care just hits Create.
   * Response shape: `{cwd: string}`.
   */
  getDefaultCwd: () => request('/api/fs/default-cwd'),

  /* -------------------- P2P (LAN pairing + public pool) ----------------- */
  /** This install's identity (device_id, label, public key). */
  p2pIdentity: () => request('/api/p2p/identity'),

  /** Rename the local device (label only — keypair is unchanged). */
  p2pSetLabel: (label) =>
    request('/api/p2p/identity', {
      method: 'PATCH',
      body: JSON.stringify({ label }),
    }),

  /** Snapshot of LAN peers currently advertising via mDNS. */
  p2pDiscover: () => request('/api/p2p/discover'),

  /** Generate a fresh PIN to display on this device. */
  p2pPairStart: () =>
    request('/api/p2p/pair/start', { method: 'POST' }),

  /** Cancel a pending pairing offer. */
  p2pPairCancel: (pairingId) =>
    request(`/api/p2p/pair/${pairingId}`, { method: 'DELETE' }),

  /** List currently-active pairing offers (for UI restore on refresh). */
  p2pPairPending: () => request('/api/p2p/pair/pending'),

  /**
   * Build a signed pairing claim FROM this device's identity.
   * Used when the user is on this device and types the PIN displayed
   * on the host. Returns the blob to POST to the host's accept endpoint.
   */
  p2pPairBuildClaim: (pin, nonce, hostPublicKey) =>
    request('/api/p2p/pair/build-claim', {
      method: 'POST',
      body: JSON.stringify({
        pin,
        nonce,
        host_public_key_b64: hostPublicKey,
      }),
    }),

  /**
   * Accept a pairing claim (host side). Receives the signed proof
   * from the claimant, verifies, persists trust anchor.
   */
  p2pPairAccept: (body) =>
    request('/api/p2p/pair/accept', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** List paired devices. */
  p2pListPaired: () => request('/api/p2p/paired'),

  /** Remove a pairing record (this side only). */
  p2pUnpair: (deviceId) =>
    request(`/api/p2p/paired/${encodeURIComponent(deviceId)}`, {
      method: 'DELETE',
    }),

  /** Read the public-pool opt-in state. Default: enabled. */
  p2pPublicPoolStatus: () => request('/api/p2p/public-pool'),

  /** Toggle the public-pool opt-in. */
  p2pPublicPoolSet: (enabled) =>
    request('/api/p2p/public-pool', {
      method: 'PATCH',
      body: JSON.stringify({ enabled: !!enabled }),
    }),

  /** Live rendezvous loop status — STUN candidates + last register/heartbeat. */
  p2pRendezvousStatus: () => request('/api/p2p/rendezvous/status'),
}
