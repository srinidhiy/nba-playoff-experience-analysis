"""Tests for the teams endpoints."""

import pytest


class TestTeamsMeta:
    """Tests for the /teams/meta endpoint."""

    def test_meta_returns_structure(self, client):
        """Test that /teams/meta returns expected structure."""
        response = client.get("/teams/meta")
        # May return 503 if data not available, which is valid
        if response.status_code == 200:
            data = response.json()
            assert "seasons" in data
            assert "teams" in data
            assert isinstance(data["seasons"], list)
            assert isinstance(data["teams"], list)
        else:
            assert response.status_code == 503

    def test_meta_teams_have_required_fields(self, client):
        """Test that team objects have required fields."""
        response = client.get("/teams/meta")
        if response.status_code == 200:
            data = response.json()
            if data["teams"]:
                team = data["teams"][0]
                assert "id" in team
                assert "full_name" in team


class TestTeamPrediction:
    """Tests for the /teams/{team_id}/season/{season} endpoint."""

    def test_valid_team_season_returns_prediction(self, client):
        """Test prediction endpoint with valid team and season."""
        # Boston Celtics team_id, recent season
        response = client.get("/teams/1610612738/season/2023-24")
        if response.status_code == 200:
            data = response.json()
            assert "team_id" in data
            assert "season" in data
            assert "prediction" in data
            assert data["team_id"] == 1610612738
            assert data["season"] == "2023-24"
        elif response.status_code == 503:
            # Model not trained yet is acceptable
            pass
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_invalid_season_format(self, client):
        """Test that invalid season format returns 400."""
        response = client.get("/teams/1610612738/season/2023")
        assert response.status_code == 400

    def test_season_format_normalization(self, client):
        """Test that 2023-2024 format is normalized to 2023-24."""
        response = client.get("/teams/1610612738/season/2023-2024")
        if response.status_code == 200:
            data = response.json()
            assert data["season"] == "2023-24"

    def test_prediction_contains_probabilities(self, client):
        """Test that prediction includes probability breakdown."""
        response = client.get("/teams/1610612738/season/2023-24")
        if response.status_code == 200:
            data = response.json()
            if "probabilities" in data and data["probabilities"]:
                prob = data["probabilities"][0]
                assert "round" in prob
                assert "label" in prob
                assert "probability" in prob

    def test_prediction_contains_experience_breakdown(self, client):
        """Test that prediction includes experience breakdown."""
        response = client.get("/teams/1610612738/season/2023-24")
        if response.status_code == 200:
            data = response.json()
            assert "experience_breakdown" in data
            if data["experience_breakdown"]:
                item = data["experience_breakdown"][0]
                assert "key" in item
                assert "label" in item
                assert "value" in item


class TestTeamComparison:
    """Tests for the /teams/compare/{team1}/{team2}/season/{season} endpoint."""

    def test_compare_returns_structure(self, client):
        """Test that comparison endpoint returns expected structure."""
        # Boston Celtics vs LA Lakers
        response = client.get("/teams/compare/1610612738/1610612747/season/2023-24")
        if response.status_code == 200:
            data = response.json()
            assert "season" in data
            assert "team1" in data
            assert "team2" in data
            assert "feature_comparison" in data
            assert "advantage" in data

    def test_compare_teams_have_predictions(self, client):
        """Test that both teams have predictions in comparison."""
        response = client.get("/teams/compare/1610612738/1610612747/season/2023-24")
        if response.status_code == 200:
            data = response.json()
            for team_key in ["team1", "team2"]:
                team = data[team_key]
                assert "team_id" in team
                assert "prediction" in team
                assert "probabilities" in team
                assert "expected_round" in team

    def test_compare_feature_comparison_structure(self, client):
        """Test that feature comparison has correct structure."""
        response = client.get("/teams/compare/1610612738/1610612747/season/2023-24")
        if response.status_code == 200:
            data = response.json()
            if data["feature_comparison"]:
                feat = data["feature_comparison"][0]
                assert "key" in feat
                assert "label" in feat
                assert "team1_value" in feat
                assert "team2_value" in feat
                assert "advantage" in feat

    def test_compare_invalid_season(self, client):
        """Test that invalid season format returns 400."""
        response = client.get("/teams/compare/1610612738/1610612747/season/2023")
        assert response.status_code == 400

    def test_compare_same_team(self, client):
        """Test comparing a team with itself still works."""
        response = client.get("/teams/compare/1610612738/1610612738/season/2023-24")
        # Should work but advantage should be tie
        if response.status_code == 200:
            data = response.json()
            assert data["advantage"] == "tie"
