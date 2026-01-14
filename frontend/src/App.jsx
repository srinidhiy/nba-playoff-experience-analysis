import { useEffect, useState } from "react";

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

function ProbabilityChart({ data }) {
  const maxValue = Math.max(...data.map((item) => item.probability), 1);
  return (
    <div className="chart">
      {data.map((item) => (
        <div key={item.round} className="chart-row">
          <span className="chart-label">{item.label}</span>
          <div className="chart-bar">
            <span
              className="chart-fill"
              style={{ width: `${(item.probability / maxValue) * 100}%` }}
            />
          </div>
          <span className="chart-value">
            {(item.probability * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [teamId, setTeamId] = useState("");
  const [season, setSeason] = useState(defaultSeason);
  const [teams, setTeams] = useState([]);
  const [seasons, setSeasons] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [metaLoading, setMetaLoading] = useState(true);

  const loadMeta = async () => {
    setMetaLoading(true);
    try {
      const response = await fetch("/teams/meta");
      if (!response.ok) {
        throw new Error("Meta request failed");
      }
      const payload = await response.json();
      setTeams(payload.teams ?? []);
      setSeasons(payload.seasons ?? []);
      if (!teamId && payload.teams?.length) {
        setTeamId(String(payload.teams[0].id));
      }
      if (payload.seasons?.length) {
        setSeason(payload.seasons[payload.seasons.length - 1]);
      }
    } catch (err) {
      setError("Unable to load teams/seasons metadata.");
    } finally {
      setMetaLoading(false);
    }
  };

  useEffect(() => {
    loadMeta();
  }, []);

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
          Model-based projections using experience, age, and prior playoff depth.
        </p>
        <p className="note">
          For past seasons, this is a retrospective model estimate using that
          season’s features.
        </p>
      </header>

      <section className="panel">
        <form onSubmit={handleSubmit} className="form">
          <label>
            Team
            <select
              value={teamId}
              onChange={(event) => setTeamId(event.target.value)}
              disabled={metaLoading}
            >
              <option value="" disabled>
                {metaLoading ? "Loading teams..." : "Select a team"}
              </option>
              {teams.map((team) => (
                <option key={team.id} value={team.id}>
                  {team.full_name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Season
            <select
              value={season}
              onChange={(event) => setSeason(event.target.value)}
              disabled={metaLoading}
            >
              <option value="" disabled>
                {metaLoading ? "Loading seasons..." : "Select a season"}
              </option>
              {seasons.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
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
                  <strong>{result.prediction?.label ?? "TBD"}</strong>
                </p>
                <p>
                  Round id: <strong>{result.prediction?.round ?? "-"}</strong>
                </p>
                <p className="note">{result.notes}</p>
                {result.probabilities && (
                  <>
                    <h3>Probability breakdown</h3>
                    <ProbabilityChart data={result.probabilities} />
                    <table className="summary-table">
                      <thead>
                        <tr>
                          <th>Round</th>
                          <th>Probability</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.probabilities.map((item) => (
                          <tr key={`${item.round}-prob`}>
                            <td>{item.label}</td>
                            <td>{(item.probability * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}
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
