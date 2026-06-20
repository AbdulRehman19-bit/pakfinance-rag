import { useState } from "react";
import SearchBar from "../components/SearchBar";
import AnswerPanel from "../components/AnswerPanel";
import { askQuery } from "../api/client";

export default function Home() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleAsk(query) {
    setIsLoading(true);
    setError(null);
    try {
      const data = await askQuery(query);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <header className="page-header">
        <p className="eyebrow">PSX · SBP · FBR</p>
        <h1 className="wordmark">PakFinance RAG</h1>
        <p className="tagline">A registry search over Pakistan's financial and regulatory record.</p>
      </header>

      <SearchBar onAsk={handleAsk} isLoading={isLoading} />
      <AnswerPanel result={result} error={error} isLoading={isLoading} />
    </main>
  );
}