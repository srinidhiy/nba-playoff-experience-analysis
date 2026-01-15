import { useEffect, useState } from "react";

const defaultSeason = "2025-26";

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

function ComparisonBar({ label, team1Value, team2Value, team1Name, team2Name, lowerIsBetter = false }) {
  const total = team1Value + team2Value || 1;
  const team1Pct = (team1Value / total) * 100;
  const team2Pct = (team2Value / total) * 100;
  
  let team1Better, team2Better;
  if (lowerIsBetter) {
    team1Better = team1Value < team2Value;
    team2Better = team2Value < team1Value;
  } else {
    team1Better = team1Value > team2Value;
    team2Better = team2Value > team1Value;
  }

  return (
    <div className="comparison-row">
      <span className={`comparison-value left ${team1Better ? 'winner' : ''}`}>
        {team1Value.toFixed(1)}
      </span>
      <div className="comparison-bar-container">
        <div className="comparison-bar">
          <span
            className={`comparison-fill left ${team1Better ? 'winner' : ''}`}
            style={{ width: `${team1Pct}%` }}
          />
          <span
            className={`comparison-fill right ${team2Better ? 'winner' : ''}`}
            style={{ width: `${team2Pct}%` }}
          />
        </div>
        <span className="comparison-label">{label}</span>
      </div>
      <span className={`comparison-value right ${team2Better ? 'winner' : ''}`}>
        {team2Value.toFixed(1)}
      </span>
    </div>
  );
}

function AllTeamsPredictions({ teams, season }) {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const loadAllPredictions = async () => {
    if (!teams.length || !season) return;
    setLoading(true);
    try {
      const results = await Promise.all(
        teams.map(async (team) => {
          try {
            const res = await fetch(`/teams/${team.id}/season/${season}`);
            if (!res.ok) return null;
            const data = await res.json();
            const finalsProb = data.probabilities?.find(p => p.round === 4)?.probability || 0;
            return {
              id: team.id,
              name: team.full_name,
              abbrev: team.abbreviation,
              prediction: data.prediction?.label || "N/A",
              finalsProb: finalsProb * 100,
              winPct: (data.features_used?.team_win_pct || 0) * 100,
              netRating: data.features_used?.net_rating || 0,
              seed: data.features_used?.seed || 0,
              avgPlayoffGames: data.features_used?.avg_playoff_games_prior || 0,
            };
          } catch {
            return null;
          }
        })
      );
      const valid = results.filter(Boolean).sort((a, b) => b.finalsProb - a.finalsProb);
      setPredictions(valid);
      setLoaded(true);
    } catch (err) {
      console.error("Failed to load predictions", err);
    } finally {
      setLoading(false);
    }
  };

  if (!loaded) {
    return (
      <div className="featured-predictions">
        <h2>{season} Playoff Predictions</h2>
        <button onClick={loadAllPredictions} disabled={loading || !teams.length}>
          {loading ? "Loading all teams..." : "Load All Team Predictions"}
        </button>
      </div>
    );
  }

  return (
    <div className="featured-predictions">
      <h2>{season} Playoff Predictions</h2>
      <p className="note">Sorted by Finals probability. Click any team for details.</p>
      <div className="predictions-table-wrapper">
        <table className="predictions-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>Predicted Round</th>
              <th>Finals %</th>
              <th>Win %</th>
              <th>Net Rtg</th>
              <th>Seed</th>
              <th>Exp</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((team, idx) => (
              <tr key={team.id} className={idx < 4 ? "top-contender" : ""}>
                <td>{idx + 1}</td>
                <td>
                  <span className="team-name">{team.name}</span>
                  <span className="team-abbrev-small">{team.abbrev}</span>
                </td>
                <td className="predicted-round-cell">{team.prediction}</td>
                <td className="finals-prob">{team.finalsProb.toFixed(1)}%</td>
                <td>{team.winPct.toFixed(1)}%</td>
                <td className={team.netRating >= 0 ? "positive" : "negative"}>
                  {team.netRating >= 0 ? "+" : ""}{team.netRating.toFixed(1)}
                </td>
                <td>{team.seed > 0 ? team.seed : "-"}</td>
                <td>{team.avgPlayoffGames.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonResult({ data }) {
  if (!data) return null;

  const team1 = data.team1;
  const team2 = data.team2;

  return (
    <div className="comparison-result">
      <div className="comparison-header">
        <div className={`team-card ${data.advantage === 'team1' ? 'advantage' : ''}`}>
          <h3>{team1.full_name}</h3>
          <p className="prediction-label">{team1.prediction.label}</p>
          <p className="expected-round">Expected: {team1.expected_round.toFixed(2)}</p>
        </div>
        <div className="vs-badge">VS</div>
        <div className={`team-card ${data.advantage === 'team2' ? 'advantage' : ''}`}>
          <h3>{team2.full_name}</h3>
          <p className="prediction-label">{team2.prediction.label}</p>
          <p className="expected-round">Expected: {team2.expected_round.toFixed(2)}</p>
        </div>
      </div>

      <h3>Feature Comparison</h3>
      <div className="comparison-bars">
        {data.feature_comparison.map((feat) => (
          <ComparisonBar
            key={feat.key}
            label={feat.label}
            team1Value={feat.team1_value}
            team2Value={feat.team2_value}
            team1Name={team1.full_name}
            team2Name={team2.full_name}
            lowerIsBetter={feat.key === 'seed'}
          />
        ))}
      </div>

      <div className="probability-comparison">
        <div className="prob-column">
          <h4>{team1.abbreviation || team1.full_name}</h4>
          <ProbabilityChart data={team1.probabilities} />
        </div>
        <div className="prob-column">
          <h4>{team2.abbreviation || team2.full_name}</h4>
          <ProbabilityChart data={team2.probabilities} />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [viewMode, setViewMode] = useState("single"); // 'single' or 'compare'
  const [teamId, setTeamId] = useState("");
  const [team1Id, setTeam1Id] = useState("");
  const [team2Id, setTeam2Id] = useState("");
  const [season, setSeason] = useState(defaultSeason);
  const [teams, setTeams] = useState([]);
  const [seasons, setSeasons] = useState([]);
  const [result, setResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
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
        setTeam1Id(String(payload.teams[0].id));
        if (payload.teams.length > 1) {
          setTeam2Id(String(payload.teams[1].id));
        }
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
        let message = "Request failed";
        try {
          const payload = await response.json();
          if (payload?.detail) {
            message = payload.detail;
          }
        } catch (err) {
          // Ignore response parsing errors.
        }
        throw new Error(message);
      }
      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError(err.message || "Unable to fetch prediction yet. Check backend status.");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    setComparisonResult(null);
    try {
      const response = await fetch(`/teams/compare/${team1Id}/${team2Id}/season/${season}`);
      if (!response.ok) {
        let message = "Request failed";
        try {
          const payload = await response.json();
          if (payload?.detail) {
            message = payload.detail;
          }
        } catch (err) {
          // Ignore response parsing errors.
        }
        throw new Error(message);
      }
      const payload = await response.json();
      setComparisonResult(payload);
    } catch (err) {
      setError(err.message || "Unable to fetch comparison. Check backend status.");
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
          season's features.
        </p>
      </header>

      <AllTeamsPredictions teams={teams} season={season} />

      <div className="mode-toggle">
        <button
          className={viewMode === "single" ? "active" : ""}
          onClick={() => setViewMode("single")}
        >
          Single Team
        </button>
        <button
          className={viewMode === "compare" ? "active" : ""}
          onClick={() => setViewMode("compare")}
        >
          Compare Teams
        </button>
      </div>

      <section className="panel">
        {viewMode === "single" ? (
          <>
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
          </>
        ) : (
          <>
            <form onSubmit={handleCompare} className="form compare-form">
              <div className="compare-teams">
                <label>
                  Team 1
                  <select
                    value={team1Id}
                    onChange={(event) => setTeam1Id(event.target.value)}
                    disabled={metaLoading}
                  >
                    <option value="" disabled>
                      {metaLoading ? "Loading..." : "Select team"}
                    </option>
                    {teams.map((team) => (
                      <option key={team.id} value={team.id}>
                        {team.full_name}
                      </option>
                    ))}
                  </select>
                </label>
                <span className="vs-text">vs</span>
                <label>
                  Team 2
                  <select
                    value={team2Id}
                    onChange={(event) => setTeam2Id(event.target.value)}
                    disabled={metaLoading}
                  >
                    <option value="" disabled>
                      {metaLoading ? "Loading..." : "Select team"}
                    </option>
                    {teams.map((team) => (
                      <option key={team.id} value={team.id}>
                        {team.full_name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
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
              <button type="submit" disabled={loading || team1Id === team2Id}>
                {loading ? "Loading..." : "Compare Teams"}
              </button>
              {team1Id === team2Id && team1Id && (
                <p className="note">Select two different teams to compare</p>
              )}
            </form>

            {error && <p className="error">{error}</p>}
            <ComparisonResult data={comparisonResult} />
          </>
        )}
      </section>
    </div>
  );
}
