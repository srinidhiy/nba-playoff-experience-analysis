import { useState } from "react";

const defaultSeason = "2023-24";

const defaultBreakdown = [
  { key: "avg_age", label: "Avg age", value: 27.4 },
  { key: "avg_seasons", label: "Avg seasons in league", value: 5.6 },
  { key: "avg_playoff_games", label: "Avg playoff games", value: 42.0 },
  { key: "injury_games", label: "Injury games missed", value: 118.0 }
];

function ExperienceChart({ data }) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="chart">
      {data.map((item) => (
        <div key={item.key} className="chart-row">
          <span className="chart-label">{item.label}</span>
          <div className="chart-bar">
            <span
              className="chart-fill"
              style={{ width: `${(item.value / maxValue) * 100}%` }}
            />
          </div>
          <span className="chart-value">{item.value.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [teamId, setTeamId] = useState("1610612747");
  const [season, setSeason] = useState(defaultSeason);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const response = await fetch(`/teams/${teamId}/season/${season}`);
      if (!response.ok) {
        throw new Error("Request failed");
      }
      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError("Unable to fetch prediction yet. Check backend status.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <h1>Playoff Experience Predictor</h1>
        <p>
          Explore how roster experience, age, and playoff depth relate to
          postseason outcomes.
        </p>
      </header>

      <section className="panel">
        <form onSubmit={handleSubmit} className="form">
          <label>
            Team ID
            <input
              value={teamId}
              onChange={(event) => setTeamId(event.target.value)}
              placeholder="1610612747"
            />
          </label>
          <label>
            Season
            <input
              value={season}
              onChange={(event) => setSeason(event.target.value)}
              placeholder="2023-24"
            />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Loading..." : "Get prediction"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}
        {result && (
          <div className="result">
            <h2>Prediction</h2>
            <div className="result-grid">
              <div>
                <p>
                  Projected round:{" "}
                  <strong>{result.prediction?.playoff_round ?? "TBD"}</strong>
                </p>
                <p>
                  Confidence:{" "}
                  <strong>{result.prediction?.confidence ?? 0}</strong>
                </p>
                <p className="note">{result.notes}</p>
              </div>
              <div>
                <h3>Experience snapshot</h3>
                <ExperienceChart
                  data={result.experience_breakdown ?? defaultBreakdown}
                />
                <table className="summary-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(result.experience_breakdown ?? defaultBreakdown).map(
                      (item) => (
                        <tr key={`${item.key}-row`}>
                          <td>{item.label}</td>
                          <td>{item.value.toFixed(1)}</td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
