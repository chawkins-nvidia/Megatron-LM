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
- retain rank 0 as the sole W&B owner and use a single W&B run ID across
  restart jobs.

Megatron reads its runtime source identity directly from verified git `HEAD`;
it does not accept a source-commit override. Tier 1 and Tier 2 are intentionally
not advertised. A request above Tier 0 or for a runtime field absent from the
capability document must remain a pre-submit failure.

The approved artifact schema currently has no explicit top-level fields for the
capability-file hash or the combined schema hash. Megatron binds both into the
consensus descriptor digest while retaining the schema's exact field set. If
Scaling requires those identities as separately addressable manifest fields,
Scaling must first version and approve an artifact-schema change; Megatron must
not add undeclared fields to `diag/v2/artifact`.
