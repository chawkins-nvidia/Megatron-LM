# diag/v2 Tier-0 launcher compatibility

Megatron advertises the integrated heartbeat through the root
`.diag_v2_runtime_capabilities.json`. The Scaling launcher must hash the two
packaged schemas using `filename + NUL + raw bytes + NUL`, require schema hash
`7b7e156949da5370cccf5fd825784dfcc4de79b19ee4eca1d79d976783b5b28a`, and
compare the capability file from the verified Megatron bundle.

The current Scaling `codex/issue-209-launch` defaults are not compatible with
this Tier-0-only implementation. Before submitting a Megatron Tier-0 run, the
launcher must make these exact changes:

- resolve `diagnostics.max_tier=0` and `diagnostics.require_tier=0`;
- pass only the seven names listed in capability `runtime_fields`, rather than
  the 48 Tier-1/Tier-2 `diag_*` and `diagnostic_*` fields in the current
  conformance overlay;
- export the already verified launch identities as
  `DIAG_V2_SCALING_COMMIT`, `DIAG_V2_RESOLVED_CONFIG_SHA256`,
  `DIAG_V2_SCALING_BUNDLE_SHA256`, and `DIAG_V2_MEGATRON_BUNDLE_SHA256`;
- retain the last world rank as Megatron's sole W&B owner and use a single W&B run ID across
  restart jobs.

Megatron reads its runtime source identity directly from verified git `HEAD`;
it does not accept a source-commit override. Tier 1 and Tier 2 are intentionally
not advertised. A request above Tier 0 or for a runtime field absent from the
capability document must remain a pre-submit failure.

The approved artifact schema currently has no explicit top-level fields for the
capability-file hash or the combined schema hash. Megatron binds both into the
consensus descriptor digest. The packaged legacy schema remains byte-exact for
its published hash; the manifest below is explicitly invalid against it and must
not be represented as conforming. Scaling must first version and approve these
artifact-schema changes before promotion.

The unchanged schema and approved validator still require Tier-0 pre/post state
snapshots, a rank-0 writer, the `global_topk_hash_v1` identity selector, and
nonempty operations for every topology process group. Megatron does not capture
state bytes or sample/token identities, preserves its last-rank writer, and
performs only three world all-reduces plus one world all-gather. It therefore
emits these events as explicitly invalid and non-promotable with unavailable
state snapshots, actual group memberships, and no invented non-world operations.

Scaling must version the sampling schema with an exact
`tier0_mask_population_checksum_v1` record containing only:

- `evidence_type`;
- `per_rank` entries with `rank`, `population`, and `checksum_value`;
- `global_population`;
- `mask_shape` in `[microbatches, micro_batch_size, cp_local_sequence]` order;
- `checksum_algorithm`;
- `collision_limitation`, explicitly stating that population plus a weighted
  checksum does not identify sample IDs, token IDs, or mask membership.

The old `global_topk_hash_v1`, selected-sample-ID, and valid-token-ID fields are
intentionally absent. Scaling must also version `digests` so it does not require
those false identity digests, and add `memory_evidence` with unavailable,
non-promotable all-rank post-gather peaks plus the sink-only post-interval sample.
Rank performance arrays are explicitly named `pre_gather_*`; the all-gather
operation carries `88 * world_size` bytes. Until that version is approved, the
current Scaling artifact gate must reject the exact three-file event.
