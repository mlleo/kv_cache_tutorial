Here's the full account of the problem and how we addressed it, in depth.
    
    The problem
    
    Where it lives. On a PD-disaggregated sglang deployment (separate prefill and decode workers), the decode server runs with --enable-hisparse. HiSparse is the hierarchical
    sparse-KV feature: instead of keeping every token's KV on the GPU, it keeps only a small per-request "device buffer" resident and swaps the top-k relevant tokens in from a
    larger host (CPU DRAM) pool on demand.
    
    When a request finishes prefill and is handed to the decode worker, HiSparseCoordinator.admit_request_direct runs. If the request's sequence is short enough to fit the device
    buffer (seq_len <= device_buffer_size), the code eagerly preloads the request's entire KV from the host pool into that device buffer, so the decode loop never has to go to
    host memory for it. That eager load is _preload_to_device_buffer.
    
    The inefficiency. The preload was implemented as a Python loop over every model layer:
    
    
    for layer_id in range(self.mem_pool_device.layer_num):
        self.mem_pool_host.load_to_device_per_layer(
            self.mem_pool_device, host_indices, device_locs, layer_id, io_backend="kernel",
        )
    
    
    Each iteration launches a separate host→device kernel. For GLM-5.2 (78 layers) that's 78 kernel launches per admitted request. These launches are enqueued back-to-back on the
    scheduler's default stream — there's no stream context or event machinery around the preload (unlike the staging path, which is explicitly overlapped on write_staging_stream).
    So 78 serialized GPU submissions sit on the admission critical path, ahead of the request's first decode step. The feature's own author flagged it: `TODO(hzh0425): Optimize
    this.`
    
    Why it was safe to fix. The reverse direction already had the batched form. MLATokenToKVPoolHost.backup_from_device_all_layer (device→host) moves ALL layers in ONE
    transfer_kv_all_layer_mla kernel call (or a JIT equivalent), and it's already used on the same pool objects with the same index tensors in admit_request_into_staging. There
    was simply no host→device counterpart. So the fix is a direction-reversal of an existing, proven API — not new machinery.
    
    Classification: algorithmic, not micro-optimization. It replaces N kernel launches with 1 — a complexity-class reduction in launch/submission count, with identical bytes
    moved. That's the kind of change that survives a "premature optimization" challenge, unlike vectorizing a small Python loop where GPU work dominates.
    
    The solution, and how it was built
    
    The change is 4 stacked commits, Python-only, no CUDA/kernel changes:
    
    1. pool_host/base.py — a concrete default. Added load_to_device_all_layer to HostKVCache as a CONCRETE method whose body is the per-layer loop. Deliberately NOT
    @abc.abstractmethod: HostKVCache has 8+ subclasses (MHA, Mamba, DSA-indexer, DeepSeekV4State, unified, etc.), and an abstractmethod would break every one that doesn't
    override it. The concrete default keeps all of them working unchanged; only the subclass with a batched kernel overrides it.
    
    2. pool_host/mla.py — the batched override. MLATokenToKVPoolHost.load_to_device_all_layer, written as an exact mirror of backup_from_device_all_layer with src/dst
    swapped for the host→device direction. The critical work was getting kernel directionality right per layout/io_backend:
    - layer_first + kernel: the all-layer kernels (transfer_kv_all_layer_mla, JIT transfer_hicache_all_layer_mla) are symmetric memcpy-style — swapping src/dst reverses
    direction. This is the ONE branch that gets the N→1 win. dst=device ptrs, src=host ptrs; dst stride uses device_row_stride_bytes.
    - layer_first + direct: transfer_kv_direct is likewise symmetric.
    - kernel_ascend (NPU): transfer_kv_dim_exchange takes an explicit direction=TransferDirection.H2D — fully symmetric, one call covers all layers.
    - page_first / page_first_direct: the all-layer kernels are directionally BAKED layer-first→page-first (device→host for backup); transfer_kv_all_layer_mla_lf_pf and the
    staged write-back JIT kernel cannot be reversed by argument swapping, and no page-first→layer-first all-layer counterpart exists. So for these layouts the code falls back to the
    per-layer H2D loop (transfer_kv_per_layer_mla_pf_lf, which IS the correct direction). No wrong-direction kernel is ever called.
    - Layer-sharded (CP) device pools: per-layer fallback over owned layer ids + MTP draft pools, mirroring backup.
    
    3. managers/hisparse_coordinator.py — the switch. _preload_to_device_buffer now calls load_to_device_all_layer once. The M3/MiMaha branch keeps its own per-layer loop
    because HiSparseMHATokenToKVPoolHost has no batched override (it inherits the base-class default, which is itself the per-layer loop).
    
    4. pool_host/mla.py (the review fix) — DCP double-translation. During review, a real bug was caught and fixed. load_to_device_all_layer translates indices via
    maybe_dcp_kernel_indices at entry; its fallback loops were then calling the PUBLIC load_to_device_per_layer, which translates AGAIN — silently corrupting indices when
    dcp_size > 1 (decode-context-parallel). The backup path avoids this by routing through a private _backup_from_device_per_layer that assumes pre-translated indices. The fix
    mirrors that: load_to_device_per_layer was split into a thin public wrapper (ownership check + DCP translation + delegate) and a private _load_to_device_per_layer worker
    that does not re-translate. The all-layer fallback loops now call the private worker. Verified: public wrapper translates-then-delegates; the private worker never re-translates;
    translation happens exactly once on every path.
    
    Two review challenges that were investigated and resolved
    
    - "Use _resolve_device_transfer_buffers / device_pool.kv_buffer[device_layer_id]." A reviewer (reading a different/newer tree than upstream) twice claimed the code
    should use a pointer-resolver method and direct pool indexing. Verified against actual origin/main: neither _resolve_device_transfer_buffers nor device_pool.kv_buffer[...]
    exists upstream — upstream's own backup_from_device_all_layer and load_to_device_per_layer both use self.device_kv_buffers[layer_id] / self.device_data_ptrs, the exact
    form the code uses. So the implementation is the faithful in-tree mirror, and the e2e run empirically confirmed no AttributeError.
    
    Verification (what was actually run)
    
    - Static: py_compile + ast.parse on all 3 files; AST call-graph checks confirming src/dst swap correctness, the sharded fallback, the page_first per-layer fallback, zero
    directionally-wrong kernel calls, the M3 loop preserved, and correct local-layer-index convention. An 18-check ad-hoc script, all PASS.
    - Unit (branch-matched, on an H100 pod): pytest test/registered/unit/managers/test_hisparse_unit.py — 9 passed, 2 skipped.
    - Equivalence harness: one load_to_device_all_layer produces device state byte-identical to N× load_to_device_per_layer, dispatching the transfer kernel exactly once (4→1 on
    the 4-layer test pool).
    - Real E2E: a 1P1D PD pair + mini-LB router with HiSparse on decode and --hicache-mem-layout layer_first, serving glm5-mini-random. Instrumentation confirmed the preload took
    the batched layer_first / io=kernel / non-sharded branch once per admission; short-seq requests completed cleanly.
    
    Honest scope: the win is N→1 kernel launches on the admission path only (N=78 for GLM-5.2; the 2-layer test model showed 2→1, the 78 figure is structural). It requires
    layer_first host layout — the default page_first takes the per-layer fallback (correct, no speedup). No steady-state throughput claim.
    
    Current status
    
    PR #44446 is open on sgl-project/sglang (4 commits, head 27cf04b). CI is red but NOT from a code failure — every failing job traces to the call-gate / pr-gate step "Require
    run-ci label," which 403s ("Must have admin rights"). That's the external-contributor gate: the full pipeline only runs after a maintainer adds the run-ci label. The remaining
    action is to ask a maintainer for that label, and to fill the empty Speed/Accuracy sections of the PR body with the evidence above.
