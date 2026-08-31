# Native ground truth: KV Cache Manager V2 `.cpp` + binding pass

commit 4716843cee6e7a6c08bf4d8be29fae25321a9344, branch feat/native-kv-events-clean, date 2026-08-31.

Scope: this pass reads the `.cpp` implementation files under
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` and the nanobind binding
glue under `cpp/tensorrt_llm/nanobind/batch_manager/`. Prior sessions read the
Python wrapper and the headers only; this pass reads bodies.

---

## Task 1: Exact `KvCache::resize()` semantics

File read: `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp`
(function body at lines 1022–1280; helpers `_shortcutSetCapacity` at
1301–1310, `_shortcutSetHistoryLength` at 1312–~1345, `_unlockStaleBlocks` at
1462–1513, `_lockHeldBlocks` at 1515–1528, `setCapacity`/`setHistoryLength`
wrappers at 1282–1299).

### 1. Return value in the three cases — can a caller distinguish no-op from real resize?

**Verified current behavior.** The function returns `bool`, and there are
exactly two return points:

- `return true;` at `kvCache.cpp:1056` — taken when `!enableScratch` (SWA
  scratch reuse disabled) **and** both `_shortcutSetCapacity(newCap)` and
  `_shortcutSetHistoryLength(newHist)` succeed (`kvCache.cpp:1053`). This is
  the "shortcut" path and covers **both** (a) genuine no-ops (new values equal
  old values, both shortcut helpers hit their `newCap == mCapacity` /
  `newHist == mHistoryLength` early-return-true branches at `kvCache.cpp:1303`
  and `kvCache.cpp:1314`) **and** (b) real changes that happen not to cross a
  block boundary / stale-range boundary (e.g. capacity changes within the same
  block, or history length changes that don't shift any lifecycle's stale
  range — see `_shortcutSetCapacity` at `kvCache.cpp:1301-1310` and
  `_shortcutSetHistoryLength` at `kvCache.cpp:1312-1345`).
- `return true;` at `kvCache.cpp:1279` — the full/slow path completes
  successfully after actually allocating/releasing pages, resizing block
  vectors, etc. This is real case (b) when it crosses a block/stale-range
  boundary.
- `return false;` at `kvCache.cpp:1143` — the **only** failure return, taken
  when `mManager->storage().newGpuSlots(...)` throws `OutOfPagesError` during
  the slow-path allocation (`kvCache.cpp:1128-1144`).

**Conclusion:** both a genuine no-op and a real successful resize return
`true`; there is **no way for a caller to distinguish (a) from (b) via the
return value alone** — `resize()`'s `bool` return only encodes
success-vs-out-of-pages, not whether anything actually changed. (A caller
could infer "nothing changed" only by separately comparing `capacity()` /
`history_length()` before and after, which the function itself does not
expose as part of the return contract.)

### 2. What happens when `historyLength` is `nullopt`?

**Verified current behavior.** `kvCache.cpp:1028`: `int newHist =
historyLength.value_or(mHistoryLength);` — when omitted, `newHist` is set
to the *current* `mHistoryLength`, i.e. it is a no-op value fed through the
rest of the function as if the caller had explicitly passed the current
value. `historyLength.has_value()` is separately checked at
`kvCache.cpp:1032` purely to decide whether to update the
`mAvgHistoryLength` moving-average tracker — when `historyLength` is
`nullopt`, `mAvgHistoryLength.update(...)` is **not** called (only
`mAvgCapacity` may be updated if `capacity.has_value()`). So: the actual
`mHistoryLength` field is left semantically unchanged when omitted (it gets
reassigned to its own current value at `kvCache.cpp:1275`), but the
`mAvgHistoryLength` moving-average side-channel is *not* touched, whereas it
*would* be touched (with the unchanged value) if the caller explicitly
passed the current value instead of `nullopt`. This is a real, source-level
side-effect difference between "pass current value explicitly" and "omit."

### 3. State on a FAILED resize — rollback vs. partial allocation vs. atomic

**Verified current behavior.** The only failure path (`OutOfPagesError` from
`newGpuSlots`) can only be reached inside the `if (newNumBlocks >=
oldNumBlocks)` branch (`kvCache.cpp:1080`), i.e. only on a growing/no-shrink
resize. The shrink branch (`if (newNumBlocks < oldNumBlocks)`,
`kvCache.cpp:1069-1075`, which calls `_decreaseCapacity`) is mutually
exclusive with the branch that can throw, since a single call is either
shrinking or growing/equal, never both. So a failed resize never occurs after
pages have already been deallocated via `_decreaseCapacity`.

Before the allocation attempt, `_unlockStaleBlocks(newHist)` unconditionally
runs at `kvCache.cpp:1067` and returns a `backupHolders` vector of
`StaleBackup` entries recording every page it unlocked/detached
(`kvCache.cpp:1462-1513`). Scratch-slot bookkeeping
(`_takeExcessScratchSlots`) also runs unconditionally at `kvCache.cpp:1078`.

On `OutOfPagesError` (`kvCache.cpp:1139-1144`):
```
catch (OutOfPagesError const&)
{
    _recoverExcessScratchSlots(excessScratchSlots);
    _lockHeldBlocks(backupHolders);
    return false;
}
```
`_recoverExcessScratchSlots` reverses the scratch-slot bookkeeping, and
`_lockHeldBlocks(backupHolders)` (`kvCache.cpp:1515-1528`) re-locks every
page that `_unlockStaleBlocks` had unlocked, restoring `mBlocks[...]` to its
pre-call state via `batchedLockToGpu`. `mCapacity`/`mHistoryLength` are only
assigned their new values at `kvCache.cpp:1274-1275`, *after* the
allocation succeeds — so on failure they are never mutated in the first
place.

**Conclusion: the operation is effectively atomic (all-or-nothing) for the
failure case actually reachable in this code** — the explicit
recover/re-lock calls put the `KvCache` back to its pre-resize state, and no
committed field (`mCapacity`, `mHistoryLength`, `mBlocks` size) is mutated
before the point of possible failure. This is "atomic by explicit rollback,"
not "atomic by construction" — i.e., it depends on `_recoverExcessScratchSlots`
and `_lockHeldBlocks` being correct/complete inverses of the preceding
unconditional side effects; I did not verify those two helpers' bodies
line-by-line beyond what's shown (their bodies are in the same file, cited
above, and appear to be straightforward inverses of `_unlockStaleBlocks` /
`_takeExcessScratchSlots`, but a full correctness audit of every field they
touch was out of scope for this pass).

### 4. Does `resize()` ever throw rather than returning `false`?

**Verified current behavior — yes, in several documented cases, distinct from
the `OutOfPagesError`-caught path:**

- `kvCache.cpp:1036`: `throw std::invalid_argument("History length cannot be
  decreased")` if `newHist < mHistoryLength`.
- `kvCache.cpp:1038`: `throw std::invalid_argument("History length cannot
  exceed capacity")` if `newCap < newHist`.
- `kvCache.cpp:1047` (debug builds only, `TLLM_UNLIKELY(gDebug)` gate at
  1042): `TLLM_CHECK_WITH_INFO` on the SWA-scratch validity invariant —
  aborts via the `TLLM_CHECK` assertion machinery (throws
  `tensorrt_llm::common::TllmException`, not `OutOfPagesError`) if violated.
- Any other exception from `mManager->storage().newGpuSlots(...)` besides
  `OutOfPagesError` (e.g. a `CuError`/`CuOOMError` from an underlying CUDA
  driver call) is **not caught** by the `catch (OutOfPagesError const&)`
  block at `kvCache.cpp:1139` and propagates out of `resize()` uncaught. Only
  `OutOfPagesError` specifically triggers the `false`-return rollback path;
  every other exception type escapes `resize()` as a thrown exception, and in
  that case none of the rollback (`_recoverExcessScratchSlots`,
  `_lockHeldBlocks`) runs, so the object is left in whatever partially-mutated
  state existed at the throw point (i.e., NOT rolled back for non-`OutOfPagesError`
  exceptions — only the `false` path is guaranteed atomic).

So the boolean contract "`false` means out of pages, `true` means success (no
distinction from no-op)" holds only for the `OutOfPagesError` case; all other
failure modes are reported via C++ exceptions, and those specifically are
**not** covered by the rollback logic described in Q3 — that's an important
asymmetry: **`return false` is safely rolled back; an uncaught exception from
deeper in the allocator is not.**

**`setCapacity`/`setHistoryLength` convenience wrappers** (`kvCache.cpp:1282-1299`):
`setCapacity` converts a `false` return into a thrown `OutOfPagesError("Not
enough pages in GPU memory")` (`kvCache.cpp:1291`); `setHistoryLength` asserts
success via `TLLM_CHECK(success)` (`kvCache.cpp:1297`) — i.e. it treats a
`false` return (which for history-length-only resizes can only mean
`OutOfPagesError` was thrown internally) as a fatal debug-check failure, not
a recoverable condition.

### `KvCacheManager::resize(CacheLevel level, size_t quota, bool bestEfforts)` (tier-level, lower priority)

**Verified current behavior.** `kvCacheManager.cpp:420-433`:
```cpp
bool KvCacheManager::resize(CacheLevel level, size_t quota, bool bestEfforts)
{
    if (bestEfforts)
        throw std::runtime_error("best_efforts resize not implemented");
    try
    {
        _adjustLevel(level, quota);
        return true;
    }
    catch (std::exception const& e)
    {
        return false;
    }
}
```
This is a much blunter contract than the per-sequence `KvCache::resize`:
`bestEfforts=true` is simply unimplemented (throws immediately, unconditionally,
regardless of feasibility). Otherwise it delegates to `_adjustLevel(level,
quota)` and converts **any** `std::exception` (not just an
out-of-pages-flavored one) into a `false` return, swallowing the specific
exception type/message entirely. I did not trace `_adjustLevel`'s body for
rollback/atomicity guarantees (out of scope/lower priority per the task); that
would need a further read of `_adjustLevel` in `kvCacheManager.cpp` /
`storageManager.cpp` to know whether a failed `_adjustLevel` leaves partial
state. **Source-inconclusive, requires further reading of `_adjustLevel`** for
its rollback semantics specifically.

---

## Task 2: Process-terminating vs. Python-catchable exceptions

### 1. Which methods are wrapped by `terminateOnException`?

**Verified current behavior.** A repo-wide grep for `terminateOnException`
under `kv_cache_manager_v2/` and the nanobind binding directory found exactly
**one** call site in the entire codebase:

`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/stagingBuffer.cpp:160`,
inside `StagingBufferManager::~StagingBufferManager() noexcept`
(`stagingBuffer.cpp:157-169`):
```cpp
StagingBufferManager::~StagingBufferManager() noexcept
{
    terminateOnException("Failed to destroy staging-buffer manager safely",
        [&]()
        {
            ...
            TLLM_CHECK_WITH_INFO(range.retired, "Destroying a staging-buffer manager with a live buffer");
            synchronizeAll(readyEvents);
        });
}
```
This guards only the destructor body of `StagingBufferManager` (a low-level
host-staging-buffer helper, not `KvCache` or `KvCacheManager` themselves).
**No public `KvCacheManager` or `KvCache` method** (`resize`, `suspend`,
`createKvCache`, `resume`, `commit`, etc.) is wrapped by
`terminateOnException` anywhere in the `.cpp` files read this session. Every
other public method lets exceptions propagate normally. Separately,
`KvCacheManager::~KvCacheManager()` (`kvCacheManager.cpp:130-140`) catches
`std::exception` around `shutdown()` and only logs via `TLLM_LOG_ERROR` —
it does **not** call `std::terminate()` or rethrow, so it is not a
`terminateOnException`-style hard-abort guard; it's a log-and-swallow
destructor guard, distinct in behavior from `terminateOnException`.

### 2. Default nanobind exception translation behavior

**Verified current behavior.** In
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:787-849`, the
module explicitly registers Python exception classes and translators for the
whole `kv_cache_manager_v2` exception hierarchy:
```cpp
static nb::object sOutOfMemoryError = nb::exception<kv::OutOfMemoryError>(m, "OutOfMemoryError");   // :787
static nb::object sHostOOMError     = nb::exception<kv::HostOOMError>(m, "HostOOMError", sOutOfMemoryError); // :788
static nb::object sDiskOOMError     = nb::exception<kv::DiskOOMError>(m, "DiskOOMError", sOutOfMemoryError); // :789
static nb::object sCuOOMError       = nb::exception<kv::CuOOMError>(m, "CuOOMError", sOutOfMemoryError);     // :790
static nb::object sLogicError       = nb::exception<kv::LogicError>(m, "LogicError");                        // :791
static nb::object sResourceBusyError = nb::exception<kv::ResourceBusyError>(m, "ResourceBusyError");         // :792
static nb::object sOutOfPagesError  = nb::exception<kv::OutOfPagesError>(m, "OutOfPagesError");               // :793
static nb::object sCuError          = nb::exception<kv::CuError>(m, "CuError");                               // :794
```
plus two `nb::register_exception_translator(...)` calls (`:801` and `:835`)
that add finer translation: one attaches a numeric `error_code` attribute to
`CuError` instances (`:801-828`), and one maps `kv::AssertionError` to
Python's builtin `AssertionError` via `PyErr_SetString(PyExc_AssertionError,
...)` (`:835-847`), with an explicit comment at `exceptions.h:98-100` that
this exists specifically "so shared tests observe the same exception type as
the pure-Python backend."

nanobind's binding mechanism (`.def(...)`) always wraps the underlying C++
call so that any exception derived from `std::exception` that propagates out
of a bound function is caught and translated into a Python exception — using
the *most specific* registered translator that matches (the explicit
translators above), and falling back to nanobind's built-in default
translation (typically a generic Python `RuntimeError`) for any
`std::exception` subtype that has no explicit registration. This is
nanobind's baseline behavior for *any* function bound via `.def`, not
something `terminateOnException` provides — `terminateOnException` is the
opposite mechanism: a manual `try { ... } catch { std::terminate(); }`
wrapper used to *convert an exception into a process abort* rather than let
it be caught. **So the default behavior is "propagates as a catchable Python
exception"; `terminateOnException` is the deliberate, narrowly-scoped
exception (used exactly once, in a destructor, per Q1).** This matches the
explicit registration of `OutOfMemoryError`, `HostOOMError`, `DiskOOMError`,
`CuOOMError`, `LogicError`, `ResourceBusyError`, `OutOfPagesError`, `CuError`,
and the `AssertionError` mapping — all of the exception types declared in
`exceptions.h` have dedicated Python-side translations, confirming the intent
that these are meant to be caught in Python, not to crash the process.

### 3. Specific case: `KvCache::suspend()` / `KvCacheManager::createKvCache(...)` hitting a CUDA error or assertion failure

**Verified current behavior.** Binding sites:
- `suspend`: `cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:1686`:
  `.def("suspend", &kv::KvCache::suspend, nb::call_guard<nb::gil_scoped_release>())`
  — a plain, direct `.def` binding of the C++ member function pointer, with
  only a GIL-release call guard, no manual try/catch and no
  `terminateOnException` wrapper.
- `create_kv_cache`: `kvCacheManagerV2.cpp:2216-2230+` — bound via a lambda
  that calls `self->createKvCache(...)` directly (`kvCacheManagerV2.cpp:2222,
  2225`), again with no manual try/catch and no `terminateOnException`.

Neither `KvCache::suspend()` nor `KvCacheManager::createKvCache(...)` is
declared `noexcept` in the headers (checked: `kvCache.h:190` declares `void
suspend();` with no `noexcept`; `kvCacheManager.h:136` declares
`createKvCache(...)` with no `noexcept`).

Therefore: if either method internally hits a `CuError` (from `cuCheck` in
`exceptions.h:169-179`, e.g. a driver call failure) or an `AssertionError`
/`LogicError` (from a `TLLM_CHECK`-style assertion or `unwrap()` on a dangling
`weak_ptr`, `exceptions.h:157-164`), that exception propagates up through the
normal C++ call stack, out of the bound function, and is caught by nanobind's
standard exception-translation machinery described in Q2 — landing in Python
as a catchable `tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2.CuError`
(with `error_code` populated) or `AssertionError`, respectively. **The
process does not abort in this scenario** — `std::terminate()` is only ever
invoked from the single `terminateOnException` call site in
`StagingBufferManager::~StagingBufferManager()`, which is unrelated to
`suspend()`/`createKvCache()`'s call chains as read in this pass. (I did not
trace every function transitively called by `suspend()`/`createKvCache()` to
prove none of them ever destroys a `StagingBufferManager` synchronously on
that path; the destructor guard would only matter if a `StagingBufferManager`
happens to be destructed with a live/un-retired buffer during that call,
which is a `TLLM_CHECK_WITH_INFO` invariant violation, not an ordinary CUDA
error — so this is a very narrow, likely-unreachable edge case, not the
general answer to "does a CUDA error abort the process." Marking this
specific narrow sub-case as **source-inconclusive** since I did not do a full
transitive-call audit of every destructor reachable from `suspend()`.)

---

## Task 3: Target/draft native object independence — hidden shared state

Files read: `kvCacheManager.cpp` (constructor, `:105-128`), `storageManager.cpp`
(`StorageManager` constructor, `:260-460` range, notably `:283-285`),
`cudaVirtMem.h`/`tokenIdExt.cpp` (allocator/pool classes), and the nanobind
constructor binding in `kvCacheManagerV2.cpp:2167-2210`.

### 1. Does the constructor allocate independent CUDA memory pools per instance?

**Verified current behavior.** `KvCacheManager::KvCacheManager(...)`
(`kvCacheManager.cpp:105-128`) constructs, per instance:
- `mRadixTree = std::make_shared<BlockRadixTree>(...)` (`:116`) — owned by this
  instance's `shared_ptr`, not shared globally.
- `mStorage = std::make_shared<StorageManager>(...)` (`:119-121`) — likewise a
  fresh, independently-owned `StorageManager`.

Inside `StorageManager`'s constructor (`storageManager.cpp`, the relevant
allocator-creation line is `:283-285`):
```cpp
mGpuPhysMemAllocator = std::make_unique<PooledPhysMemAllocator>(
    CacheLevelManager::cacheTierGranularity(CacheTier::GPU_MEM, gpuQuota));
```
`mGpuPhysMemAllocator` is a `std::unique_ptr` **member** of `StorageManager`
(confirmed by its class declaration/usage; `storageManager.cpp:340, 453-456,
512, 1640-1641, 1918` all reference `mGpuPhysMemAllocator` as a per-instance
member, e.g. `.reset()` at `:512` and `->clear()` at `:1641`). `class
PooledPhysMemAllocator` (`cudaVirtMem.h:55-93`) has `mPool` as a plain
non-static `SimplePool<PhysMemWrapper>` data member and explicitly
deletes its copy constructor/assignment (`cudaVirtMem.h:64-65`) — there is no
static/global pool inside it. **Conclusion: each `KvCacheManager` instance
allocates its own independent `PooledPhysMemAllocator`, and thus its own
independent CUDA virtual-memory-backed GPU page pool; two instances do not
share a CUDA memory pool by construction.**

### 2. Does `KvCacheManager` own or share a CUDA stream, `EventSink`, or other singleton-like object?

**Verified current behavior.** The constructor signature is `explicit
KvCacheManager(KVCacheManagerConfig const& config, std::shared_ptr<EventSink>
eventSink = nullptr, ...)` (header, confirmed via
`kvCacheManager.h:107` — default value is `nullptr`, **not** a shared/global
default instance). The body simply does `mEventSink(std::move(eventSink))`
(`kvCacheManager.cpp:109`) with no fallback construction of a default
`EventSink` if the caller passes nothing — if omitted, `mEventSink` is a null
`shared_ptr`, meaning "no event sink," not "a shared global one."

This is corroborated at the Python-binding layer: the nanobind `__init__`
lambda (`kvCacheManagerV2.cpp:2168-2209`) takes `nb::object eventManager`
defaulting to `nb::none()` (`:2208`, `nb::arg("event_manager").none() =
nb::none()`), and only if the Python caller passes a non-`None` object does
it `nb::cast<std::shared_ptr<kv::EventManager>>(eventManager)` and forward it
(`:2172-2176`) — otherwise `eventSink` stays a default-constructed (null)
`shared_ptr<kv::EventSink>`. **There is no code path, in either the C++
constructor or the Python binding, where an omitted `eventSink` is silently
backed by a shared/global `EventSink` instance — each caller must
independently construct and pass its own `EventManager`, or get no event
sink at all.**

No CUDA stream is stored or shared by `KvCacheManager` at construction time
either — no `mCudaStream`-like constructor parameter or member is set in
`kvCacheManager.cpp:105-128`; streams are associated per-`KvCache` later
(`KvCache` has an `mCudaStream` member set via `activate()`/`setStream()`,
per `kvCache.cpp:128, 207, 253, 257`), which is out of scope for the manager
constructor itself but worth noting: streams are per-`KvCache`, not shared at
the manager level either.

### 3. Global/static registry, singleton, or shared allocator that two instances would both touch

**Verified current behavior — one confirmed process-global singleton found.**
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/tokenIdExt.cpp:33-56`
defines an anonymous-namespace `DigestPool` class with a Meyers-singleton
accessor:
```cpp
class DigestPool
{
public:
    static DigestPool& instance()   // tokenIdExt.cpp:54
    {
        static DigestPool pool;
        return pool;
    }
    ...
};
```
This is used by `TokenIdExt` (constructor at `tokenIdExt.cpp:206`, destructor
`:212`, copy `:217`, dereference `:222`, `liveCount()` `:230`) to store
multi-modal "digest" values (32-byte hashes) referenced by a packed 31-bit
slot index. The file's own comment block (`tokenIdExt.cpp:36-46`) explicitly
documents this as intentional: *"Digests are rare (multi-modal only), so all
access is guarded by a single mutex... a single static singleton is safe."*
This **is** genuine process-wide shared mutable state: every `TokenIdExt`
constructed by *any* `KvCacheManager`/`KvCache` instance in the process
(including a target manager and a draft manager constructed separately) reads
and writes the same `DigestPool` instance, guarded by one mutex. A bug in
digest slot allocation/freeing (e.g. a double-free or an index leak) in code
driven by one manager (e.g. the draft model's KV cache, if it processes
multi-modal tokens) could in principle corrupt or contend with digest storage
used by the other manager (e.g. the target model's KV cache) — this is a real
data-flow path where target and draft native objects are not fully isolated,
though it only activates when multi-modal digest tokens are present, and
access is documented as mutex-guarded (i.e., not a data race, but still a
shared point of contention/failure).

No other `static`/singleton/global-registry pattern touching
`KvCacheManager`, `KvCache`, `StorageManager`, or
`PooledPhysMemAllocator`-level state was found in this pass (grep for
`static` member-function/singleton patterns across all `.cpp` files under
`kv_cache_manager_v2/` turned up only `DigestPool::instance()` as a genuine
cross-instance singleton; other `static` hits were `static_cast`/`static
constexpr`/local `static` caches unrelated to instance sharing, e.g.
`blockRadixTree.cpp:88`'s `static std::string const impl =
SHA256AutoDetect();`, which is a static-initialized constant, not mutable
shared state).

### 4. Conclusion: fully independent or shared/global state?

**Verified current behavior, with one documented exception.** Target and
draft `KvCacheManager` instances, when separately constructed (as
`kv_cache_manager_v2.py`'s `self.impl = KVCacheManagerPy(config, ...)` does
per Python wrapper instance), are **independent for GPU memory pools, CUDA
virtual-memory allocators, `EventSink`/event manager, and CUDA streams** — no
default parameter or internal code path falls back to a shared/global
instance for any of these (`kvCacheManager.cpp:105-128`,
`storageManager.cpp:283-285`, `cudaVirtMem.h:55-93`,
`kvCacheManagerV2.cpp:2168-2209`).

However, they are **not fully independent in one specific, narrow respect**:
both instances' `TokenIdExt` values (used for multi-modal input token
hashing) route through the single process-global `DigestPool::instance()`
singleton at `tokenIdExt.cpp:54`, guarded by a mutex. This is deliberate,
documented, mutex-protected shared state — not a data race — but it is
nonetheless global/cross-instance state that a bug in one manager's
multi-modal digest handling could, in principle, affect the other manager
through (e.g. slot exhaustion, contention, or a logic bug in
alloc/free/duplicate bookkeeping). Aside from the GPU device itself (which is
an unavoidable OS/driver-level shared resource for any two CUDA-using
components on the same process/device) and this one `DigestPool` singleton, no
other shared mutable state between independently-constructed `KvCacheManager`
instances was found in this pass.

---

## Open Questions

1. **`_adjustLevel`'s atomicity/rollback semantics** (used by
   `KvCacheManager::resize(CacheLevel, size_t, bool)`, `kvCacheManager.cpp:426`)
   were not traced in this pass — whether a failed tier-level resize leaves
   `StorageManager` partially adjusted is unresolved. Requires reading
   `_adjustLevel`'s body (likely in `kvCacheManager.cpp` or
   `storageManager.cpp`) in a follow-up pass.
2. **`_recoverExcessScratchSlots` and `_lockHeldBlocks` full correctness as
   exact inverses** of `_takeExcessScratchSlots`/`_unlockStaleBlocks` was
   read at the call-site level and appears sound, but a field-by-field audit
   (e.g. whether `mCommitState`/`isCommitted()` bookkeeping is fully restored
   in every edge case) was not performed. Genuinely requires either a closer
   line-by-line diff of those four functions or a runtime test that forces
   the `OutOfPagesError` path and asserts full state equality before/after.
3. **Whether any transitive callee of `suspend()`/`createKvCache()` ever
   synchronously destructs a `StagingBufferManager` with a live buffer**
   (which would route through the sole `terminateOnException` call site and
   abort the process) was not exhaustively traced. This is likely
   unreachable under normal operation (it would represent a separate
   internal invariant violation, not an ordinary CUDA error), but a full
   transitive call-graph audit was out of scope here. This is a
   **source-inconclusive** point, not a proven "unreachable."
4. **Runtime confirmation of nanobind's exact default fallback exception
   type** (e.g., precisely which Python exception class a truly
   *unregistered* `std::exception` subtype would map to, if one were ever
   thrown from this module without an explicit `nb::exception<>`
   registration) was inferred from nanobind's documented/general behavior
   plus the explicit registrations found, not from directly reading
   nanobind's own translation-dispatch source in this repo. All exception
   types actually declared in `exceptions.h` do have explicit registrations,
   so this gap does not affect any concrete claim above, but it remains
   **source-inconclusive** as a fully general statement about "any possible"
   C++ exception.
