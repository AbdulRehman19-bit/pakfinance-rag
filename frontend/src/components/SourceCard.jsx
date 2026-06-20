import ProvenanceStrip from "./ProvenanceStrip";

export default function SourceCard({ chunk, index }) {
  return (
    <article className="source-card">
      <header className="source-card-header">
        <span className={`source-badge source-badge--${chunk.source?.toLowerCase()}`}>
          {chunk.source}
        </span>
        <span className="source-index">[{index + 1}]</span>
      </header>

      <h3 className="source-title">{chunk.document_title}</h3>
      {chunk.page_number ? <p className="source-page">p. {chunk.page_number}</p> : null}

      <p className="source-excerpt">{chunk.text}</p>

      <ProvenanceStrip chunk={chunk} />
    </article>
  );
}