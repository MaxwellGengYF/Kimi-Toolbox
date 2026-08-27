# KimiX LLM Configuration

This context describes how the desktop client identifies an LLM and records the exact configuration chosen for a project or session.

## Language

**Provider**:
The credential and transport boundary through which models are reached, currently a ChatGPT subscription or a Provider file.
_Avoid_: Source, service

**Provider file**:
A user-owned JSON document that completely defines one Provider-backed model configuration.
_Avoid_: JSON config, external config

**Model**:
A Provider-advertised or Provider-file-defined model identity together with its current capabilities and availability metadata.
_Avoid_: Config, reference

**Variant**:
One executable option set beneath a Model; ChatGPT reasoning effort is the first editable Variant dimension.
_Avoid_: Model alias, mode

**LLM selection**:
The secret-free pairing of a Provider target and one exact Variant stored for a project or session.
_Avoid_: Source, reference, config

**Resolved selection**:
An LLM selection matched against current Model and Variant metadata so availability can be determined without changing the stored choice.
_Avoid_: Refreshed config

## Relationships

- A Provider exposes one or more Models.
- A Model exposes one or more Variants; a Provider file exposes only `configured`.
- An LLM selection pairs exactly one Provider target with exactly one Variant key.
- A resolved selection combines that durable choice with the current model catalog.

## Rules

- A catalog default is metadata, never a durable selection.
- A saved Variant is never silently replaced when catalogs change.
- The active session owns the resolved snapshot captured at startup.
- GUI metadata contains no credential, OAuth token, or Provider request parameters.
- Explicit ChatGPT effort is applied only at the GUI subscription boundary; Kimix core
  reasoning derivation is unchanged.
