function Tick({ label, value, found }) {
  return (
    <span className={`provenance-tick ${found ? "is-found" : "is-absent"}`}>
      <span className="provenance-tick-label">{label}</span>
      <span className="provenance-tick-value">{found ? value : "—"}</span>
    </span>
  );
}

export default function ProvenanceStrip({ chunk }) {
  const { bm25_rank, dense_rank, rerank_score } = chunk;

  return (
    <div className="provenance-strip" title="How this passage was found">
      <Tick label="BM25" value={bm25_rank ? `#${bm25_rank}` : null} found={Boolean(bm25_rank)} />
      <Tick label="Dense" value={dense_rank ? `#${dense_rank}` : null} found={Boolean(dense_rank)} />
      <Tick
        label="Reranked"
        value={typeof rerank_score === "number" ? rerank_score.toFixed(3) : null}
        found={typeof rerank_score === "number"}
      />
    </div>
  );
}