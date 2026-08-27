# Boundary-normalized transcript data

**Status:** Accepted

**Date:** 2026-08-26

**Decision:** Wire envelopes and SDK messages are normalized once at the pure-layer boundary into an immutable semantic transcript AST. Historical replay and live delivery apply the same typed Transcript Mutations, and every downstream cache carries Transcript Entries or History Entries rather than flattened display text. Unknown, partial, or invalid content is retained only through explicit bounded raw fallback blocks.

**Consequences:** Tool arguments are parsed and classified in one place; layout, copying, accessibility, and Qt documents only traverse semantic blocks. History and live output converge deterministically, including result-first and trimmed-target updates. The model has more explicit data types and boundary code, but removes tuple/string dual representations and prevents display wording from becoming program logic.
