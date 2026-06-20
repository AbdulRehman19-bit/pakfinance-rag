import SourceCard from "./SourceCard";

export default function AnswerPanel({ result, error, isLoading }) {
  if (isLoading) {
    return (
      <div className="answer-panel is-loading" role="status">
        <p className="answer-status">Searching the registry…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="answer-panel is-error" role="alert">
        <p className="answer-status">Couldn't complete that search.</p>
        <p className="answer-error-detail">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="answer-panel is-empty">
        <p className="answer-status">No entry yet. Ask a question above to open one.</p>
      </div>
    );
  }

  return (
    <div className="answer-panel">
      <p className="answer-query">"{result.query}"</p>
      <p className="answer-text">{result.answer}</p>
      <p className="answer-meta">{result.latency_ms} ms · {result.sources.length} sources</p>

      {result.sources.length > 0 ? (
        <div className="sources-list">
          <h2 className="sources-heading">Filed sources</h2>
          {result.sources.map((chunk, index) => (
            <SourceCard key={chunk.chunk_id} chunk={chunk} index={index} />
          ))}
        </div>
      ) : null}
    </div>
  );
}