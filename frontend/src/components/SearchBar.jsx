import { useState } from "react";

const STARTER_QUESTIONS = [
  "What is the current SBP policy rate?",
  "What does a recent FBR SRO say about withholding tax?",
  "What are the PSX listing requirements for a new company?",
];

export default function SearchBar({ onAsk, isLoading }) {
  const [value, setValue] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    if (!value.trim() || isLoading) return;
    onAsk(value.trim());
  }

  return (
    <div className="search-panel">
      <form className="search-form" onSubmit={handleSubmit}>
        <label htmlFor="query-input" className="search-label">Ask the registry</label>
        <div className="search-input-row">
          <input
            id="query-input"
            type="text"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder="Ask about a circular, SRO, or listing rule…"
            autoComplete="off"
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !value.trim()}>
            {isLoading ? "Searching…" : "Ask"}
          </button>
        </div>
      </form>

      <div className="starter-questions" aria-label="Example questions">
        {STARTER_QUESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            className="starter-question"
            onClick={() => onAsk(question)}
            disabled={isLoading}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}