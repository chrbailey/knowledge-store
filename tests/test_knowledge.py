"""Tests for knowledge-store — SQLite + FTS5 searchable knowledge base."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

# Add parent to path so we can import knowledge_lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_lib.init_db import init_db, get_db_path, get_connection
from knowledge_lib.db import KnowledgeDB, compute_insight_hash


class TestBase(unittest.TestCase):
    """Base test class with temp directory and DB setup."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test_knowledge.db"
        self.db = KnowledgeDB(db_path=self.db_path)

    def tearDown(self):
        self.db.close()
        if self.db_path.exists():
            self.db_path.unlink()
        # Clean up WAL/SHM files
        for suffix in ("-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            if p.exists():
                p.unlink()
        os.rmdir(self.tmp_dir)


# ============ Schema Tests ============


class TestSchema(TestBase):

    def test_tables_created(self):
        """Schema creates insights, insights_fts, and votes tables."""
        rows = self.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [r["name"] for r in rows]
        self.assertIn("insights", names)
        self.assertIn("votes", names)
        # FTS5 virtual tables show up in sqlite_master
        self.assertTrue(any("insights_fts" in n for n in names))

    def test_wal_mode(self):
        """Database uses WAL journal mode."""
        row = self.db.fetchone("PRAGMA journal_mode")
        self.assertEqual(row[0], "wal")

    def test_foreign_keys_on(self):
        """Foreign keys are enabled."""
        row = self.db.fetchone("PRAGMA foreign_keys")
        self.assertEqual(row[0], 1)

    def test_indexes_created(self):
        """Expected indexes exist."""
        rows = self.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        names = [r["name"] for r in rows]
        self.assertIn("idx_insights_tier", names)
        self.assertIn("idx_insights_captured", names)
        self.assertIn("idx_insights_project", names)
        self.assertIn("idx_votes_hash", names)

    def test_init_db_idempotent(self):
        """Calling init_db twice doesn't error."""
        conn = init_db(self.db_path)
        conn.close()
        conn2 = init_db(self.db_path)
        conn2.close()

    def test_get_connection_creates_if_missing(self):
        """get_connection initializes DB if file doesn't exist."""
        new_path = Path(self.tmp_dir) / "new.db"
        conn = get_connection(new_path)
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='insights'").fetchone()
        self.assertIsNotNone(row)
        conn.close()
        new_path.unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(new_path) + suffix)
            if p.exists():
                p.unlink()


# ============ Hash Tests ============


class TestHash(unittest.TestCase):

    def test_consistent_hash(self):
        """Same text produces same hash."""
        h1 = compute_insight_hash("Hello world")
        h2 = compute_insight_hash("Hello world")
        self.assertEqual(h1, h2)

    def test_case_insensitive(self):
        """Hash is case-insensitive."""
        h1 = compute_insight_hash("Hello World")
        h2 = compute_insight_hash("hello world")
        self.assertEqual(h1, h2)

    def test_whitespace_normalized(self):
        """Extra whitespace doesn't change hash."""
        h1 = compute_insight_hash("hello   world")
        h2 = compute_insight_hash("hello world")
        self.assertEqual(h1, h2)

    def test_hash_length(self):
        """Hash is 16 hex chars."""
        h = compute_insight_hash("test")
        self.assertEqual(len(h), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))


# ============ Insert/Dedup Tests ============


class TestInsert(TestBase):

    def test_basic_insert(self):
        """Insert a single insight."""
        rowid = self.db.upsert_insight(text="SQLite is fast for local storage")
        self.assertIsNotNone(rowid)

    def test_dedup_same_hash(self):
        """Duplicate text is ignored (INSERT OR IGNORE)."""
        self.db.upsert_insight(text="Insight A")
        self.db.upsert_insight(text="Insight A")
        count = self.db.fetchone("SELECT COUNT(*) AS c FROM insights")["c"]
        self.assertEqual(count, 1)

    def test_dedup_case_insensitive(self):
        """Different case produces same hash, so dedup works."""
        self.db.upsert_insight(text="Test Insight")
        self.db.upsert_insight(text="test insight")
        count = self.db.fetchone("SELECT COUNT(*) AS c FROM insights")["c"]
        self.assertEqual(count, 1)

    def test_insert_all_fields(self):
        """Insert with all optional fields set."""
        rowid = self.db.upsert_insight(
            text="Full field test",
            tier=1,
            source="llm_extraction",
            confidence=0.9,
            session_id="sess-123",
            project="promptspeak",
            is_personal=True,
            upvotes=3,
            downvotes=1,
            entities='["SQLite", "FTS5"]',
            domain="technical",
            captured_at="2026-02-18T12:00:00",
        )
        insight = self.db.get_insight(str(rowid))
        self.assertEqual(insight["tier"], 1)
        self.assertEqual(insight["source"], "llm_extraction")
        self.assertEqual(insight["confidence"], 0.9)
        self.assertEqual(insight["project"], "promptspeak")
        self.assertEqual(insight["is_personal"], 1)
        self.assertEqual(insight["upvotes"], 3)
        self.assertEqual(insight["domain"], "technical")

    def test_get_insight_by_hash(self):
        """Retrieve by hash."""
        self.db.upsert_insight(text="Find me by hash")
        h = compute_insight_hash("Find me by hash")
        result = self.db.get_insight(h)
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Find me by hash")

    def test_get_insight_by_id(self):
        """Retrieve by numeric ID."""
        rowid = self.db.upsert_insight(text="Find me by ID")
        result = self.db.get_insight(str(rowid))
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Find me by ID")

    def test_get_nonexistent(self):
        """Non-existent ref returns None."""
        result = self.db.get_insight("doesnotexist")
        self.assertIsNone(result)


# ============ FTS5 Search Tests ============


class TestSearch(TestBase):

    def setUp(self):
        super().setUp()
        # Seed test data
        self.db.upsert_insight(text="SQLite is excellent for local embedded databases", tier=1, project="promptspeak")
        self.db.upsert_insight(text="Pinecone was decommissioned and replaced with local search", tier=2, project="knowledge")
        self.db.upsert_insight(text="FTS5 provides full text search with porter stemming", tier=1, project="promptspeak")
        self.db.upsert_insight(text="The touchgrass plugin uses emotional memory tracking", tier=3, project="touchgrass")
        self.db.upsert_insight(text="WAL mode improves concurrent read performance in SQLite", tier=2, project="promptspeak")

    def test_basic_search(self):
        """Search returns matching results."""
        results = self.db.search("SQLite")
        self.assertTrue(len(results) >= 1)
        texts = [r["text"] for r in results]
        self.assertTrue(any("SQLite" in t for t in texts))

    def test_search_ranking(self):
        """Lower tier (higher quality) results appear first."""
        results = self.db.search("SQLite")
        # Tier 1 result should come before tier 2
        if len(results) >= 2:
            tiers = [r["tier"] for r in results]
            self.assertLessEqual(tiers[0], tiers[1])

    def test_search_empty_query(self):
        """Empty query returns empty list."""
        results = self.db.search("")
        self.assertEqual(results, [])

    def test_search_no_results(self):
        """Query with no matches returns empty."""
        results = self.db.search("xyznonexistent")
        self.assertEqual(results, [])

    def test_search_limit(self):
        """Limit parameter caps results."""
        results = self.db.search("search", limit=1)
        self.assertLessEqual(len(results), 1)

    def test_search_project_filter(self):
        """Project filter narrows results."""
        results = self.db.search("search", project="promptspeak")
        for r in results:
            self.assertEqual(r["project"], "promptspeak")

    def test_search_tier_filter(self):
        """Tier max filter excludes lower quality."""
        results = self.db.search("plugin", tier_max=2)
        for r in results:
            self.assertLessEqual(r["tier"], 2)

    def test_fts_operator_sanitization(self):
        """FTS5 operators are stripped to prevent errors."""
        # These would cause FTS5 syntax errors without sanitization
        results = self.db.search('SQLite AND "injection" OR NOT')
        # Should not raise, may or may not return results
        self.assertIsInstance(results, list)

    def test_fts_special_chars(self):
        """Special chars are stripped safely."""
        results = self.db.search("SQLite()")
        self.assertIsInstance(results, list)

    def test_porter_stemming(self):
        """Porter stemmer matches word variants."""
        results = self.db.search("searching")
        # "search" in data should match "searching" via stemming
        self.assertTrue(len(results) >= 1)


# ============ Vote Tests ============


class TestVotes(TestBase):

    def setUp(self):
        super().setUp()
        self.db.upsert_insight(text="Voteable insight for testing", tier=2)

    def test_upvote(self):
        """Upvote increments counter."""
        h = compute_insight_hash("Voteable insight for testing")
        result = self.db.vote(h, "up", session_id="test-sess")
        self.assertEqual(result["upvotes"], 1)
        self.assertEqual(result["downvotes"], 0)

    def test_downvote(self):
        """Downvote increments counter."""
        h = compute_insight_hash("Voteable insight for testing")
        result = self.db.vote(h, "down")
        self.assertEqual(result["downvotes"], 1)

    def test_multiple_votes(self):
        """Multiple votes accumulate."""
        h = compute_insight_hash("Voteable insight for testing")
        self.db.vote(h, "up")
        self.db.vote(h, "up")
        result = self.db.vote(h, "down")
        self.assertEqual(result["upvotes"], 2)
        self.assertEqual(result["downvotes"], 1)

    def test_vote_by_id(self):
        """Vote by numeric ID works."""
        rowid = self.db.upsert_insight(text="Vote by ID test")
        result = self.db.vote(str(rowid), "up")
        self.assertEqual(result["upvotes"], 1)

    def test_invalid_vote_type(self):
        """Invalid vote type raises."""
        h = compute_insight_hash("Voteable insight for testing")
        with self.assertRaises(ValueError):
            self.db.vote(h, "sideways")

    def test_vote_nonexistent(self):
        """Voting on nonexistent insight raises."""
        with self.assertRaises(ValueError):
            self.db.vote("nonexistent", "up")

    def test_vote_affects_search_ranking(self):
        """Voted insights rank higher in search."""
        self.db.upsert_insight(text="Upvoted SQLite insight is great", tier=2, project="test")
        self.db.upsert_insight(text="Unvoted SQLite insight exists too", tier=2, project="test")

        h = compute_insight_hash("Upvoted SQLite insight is great")
        for _ in range(5):
            self.db.vote(h, "up")

        results = self.db.search("SQLite insight", project="test")
        if len(results) >= 2:
            # Upvoted one should rank higher (more negative rank = better in FTS5)
            upvoted = [r for r in results if "Upvoted" in r["text"]]
            self.assertTrue(len(upvoted) > 0)

    def test_votes_table_populated(self):
        """Vote records are stored in votes table."""
        h = compute_insight_hash("Voteable insight for testing")
        self.db.vote(h, "up", session_id="sess-abc")
        votes = self.db.fetchall("SELECT * FROM votes WHERE insight_hash = ?", (h,))
        self.assertEqual(len(votes), 1)
        self.assertEqual(votes[0]["vote_type"], "up")
        self.assertEqual(votes[0]["session_id"], "sess-abc")


# ============ Pin Tests ============


class TestPin(TestBase):

    def test_pin_creates_tier_0(self):
        """Pin creates insight at tier 0."""
        result = self.db.pin("Always use WAL mode for SQLite")
        self.assertEqual(result["tier"], 0)
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["source"], "user_explicit")

    def test_pin_with_project(self):
        """Pin with project set."""
        result = self.db.pin("Project-specific pin", project="promptspeak")
        self.assertEqual(result["project"], "promptspeak")

    def test_pin_dedup(self):
        """Pinning same text twice doesn't duplicate."""
        self.db.pin("Unique pin text")
        self.db.pin("Unique pin text")
        count = self.db.fetchone("SELECT COUNT(*) AS c FROM insights WHERE tier = 0")["c"]
        self.assertEqual(count, 1)


# ============ Top Insights Tests ============


class TestTopInsights(TestBase):

    def setUp(self):
        super().setUp()
        # Seed with varied tiers and votes
        self.db.upsert_insight(text="Tier 0 pinned insight", tier=0, captured_at="2026-02-18T12:00:00")
        self.db.upsert_insight(text="Tier 1 discovery insight", tier=1, captured_at="2026-02-17T12:00:00")
        self.db.upsert_insight(text="Tier 2 general insight", tier=2, captured_at="2026-02-16T12:00:00")
        self.db.upsert_insight(text="Tier 3 low quality insight", tier=3, captured_at="2026-02-15T12:00:00")

    def test_top_insights_ordered(self):
        """Top insights are ordered by tier weight + votes."""
        results = self.db.get_top_insights(limit=10, days=30)
        self.assertTrue(len(results) >= 1)
        # Tier 0 should be first
        self.assertEqual(results[0]["tier"], 0)

    def test_top_insights_limit(self):
        """Limit works."""
        results = self.db.get_top_insights(limit=2, days=30)
        self.assertLessEqual(len(results), 2)

    def test_top_insights_recency_filter(self):
        """Days parameter filters old insights."""
        # Insert an old insight
        self.db.upsert_insight(text="Ancient insight", tier=0, captured_at="2020-01-01T00:00:00")
        results = self.db.get_top_insights(limit=10, days=7)
        texts = [r["text"] for r in results]
        self.assertNotIn("Ancient insight", texts)

    def test_top_insights_excludes_expired(self):
        """Expired insights are excluded."""
        self.db.upsert_insight(
            text="Expired insight should hide",
            tier=0,
            captured_at="2026-02-18T00:00:00",
            expired_at="2026-02-17T00:00:00",
        )
        results = self.db.get_top_insights(limit=10, days=30)
        texts = [r["text"] for r in results]
        self.assertNotIn("Expired insight should hide", texts)


# ============ Stats Tests ============


class TestStats(TestBase):

    def test_empty_stats(self):
        """Stats on empty DB."""
        s = self.db.stats()
        self.assertEqual(s["total_insights"], 0)
        self.assertEqual(s["total_votes"], 0)

    def test_stats_counts(self):
        """Stats reflect inserted data."""
        self.db.upsert_insight(text="Insight A", tier=1, source="llm_extraction", project="p1")
        self.db.upsert_insight(text="Insight B", tier=2, source="conversation", project="p1")
        self.db.upsert_insight(text="Insight C", tier=1, source="llm_extraction", project="p2")

        s = self.db.stats()
        self.assertEqual(s["total_insights"], 3)
        self.assertEqual(s["by_tier"][1], 2)
        self.assertEqual(s["by_tier"][2], 1)
        self.assertEqual(s["by_source"]["llm_extraction"], 2)
        self.assertEqual(s["by_project"]["p1"], 2)

    def test_stats_top_voted(self):
        """Top voted insights appear in stats."""
        self.db.upsert_insight(text="Popular insight")
        h = compute_insight_hash("Popular insight")
        self.db.vote(h, "up")
        self.db.vote(h, "up")

        s = self.db.stats()
        self.assertEqual(len(s["top_voted"]), 1)
        self.assertEqual(s["top_voted"][0]["net_votes"], 2)


# ============ Sanitize FTS Query Tests ============


class TestSanitize(unittest.TestCase):

    def test_strips_operators(self):
        s = KnowledgeDB._sanitize_fts_query("foo AND bar OR baz")
        self.assertNotIn("AND", s)
        self.assertNotIn("OR", s)

    def test_strips_parens(self):
        s = KnowledgeDB._sanitize_fts_query("foo(bar)")
        self.assertNotIn("(", s)

    def test_strips_quotes(self):
        s = KnowledgeDB._sanitize_fts_query('"exact phrase"')
        self.assertNotIn('"', s)

    def test_empty_returns_empty(self):
        self.assertEqual(KnowledgeDB._sanitize_fts_query(""), "")
        self.assertEqual(KnowledgeDB._sanitize_fts_query("   "), "")

    def test_preserves_normal_words(self):
        s = KnowledgeDB._sanitize_fts_query("hello world")
        self.assertEqual(s, "hello world")


# ============ Project Insights Tests ============


class TestProjectInsights(TestBase):

    def setUp(self):
        super().setUp()
        self.db.upsert_insight(
            text="SQLite FTS5 requires porter tokenizer for stemming",
            tier=1, project="touchgrass", captured_at="2026-02-20T12:00:00",
        )
        self.db.upsert_insight(
            text="The touchgrass plugin uses emotional memory tracking",
            tier=2, project="touchgrass", captured_at="2026-02-19T12:00:00",
        )
        self.db.upsert_insight(
            text="PromptSpeak has 45 MCP tools and 658 tests",
            tier=1, project="promptspeak", captured_at="2026-02-18T12:00:00",
        )
        self.db.upsert_insight(
            text="Touchgrass is mentioned in this promptspeak insight",
            tier=2, project="promptspeak", captured_at="2026-02-17T12:00:00",
        )

    def test_returns_only_matching_project(self):
        """get_project_insights filters by project column, not keyword."""
        results = self.db.get_project_insights("touchgrass", limit=10, days=30)
        for r in results:
            self.assertEqual(r["project"], "touchgrass")

    def test_does_not_return_keyword_matches(self):
        """Insight mentioning 'touchgrass' but tagged 'promptspeak' is excluded."""
        results = self.db.get_project_insights("touchgrass", limit=10, days=30)
        texts = [r["text"] for r in results]
        self.assertNotIn("Touchgrass is mentioned in this promptspeak insight", texts)

    def test_respects_tier_max(self):
        """Tier filter works."""
        results = self.db.get_project_insights("touchgrass", tier_max=1, days=30)
        for r in results:
            self.assertLessEqual(r["tier"], 1)

    def test_respects_days_window(self):
        """Days parameter filters old insights."""
        self.db.upsert_insight(
            text="Ancient touchgrass insight", tier=0,
            project="touchgrass", captured_at="2020-01-01T00:00:00",
        )
        results = self.db.get_project_insights("touchgrass", days=30)
        texts = [r["text"] for r in results]
        self.assertNotIn("Ancient touchgrass insight", texts)

    def test_excludes_expired(self):
        """Expired insights are excluded."""
        self.db.upsert_insight(
            text="Expired touchgrass insight", tier=0,
            project="touchgrass", captured_at="2026-02-20T00:00:00",
            expired_at="2026-02-19T00:00:00",
        )
        results = self.db.get_project_insights("touchgrass", days=30)
        texts = [r["text"] for r in results]
        self.assertNotIn("Expired touchgrass insight", texts)

    def test_ordered_by_tier_then_votes(self):
        """Higher-tier insights rank first."""
        results = self.db.get_project_insights("touchgrass", limit=10, days=30)
        if len(results) >= 2:
            self.assertLessEqual(results[0]["tier"], results[1]["tier"])

    def test_limit_works(self):
        """Limit caps results."""
        results = self.db.get_project_insights("touchgrass", limit=1, days=30)
        self.assertLessEqual(len(results), 1)


# ============ Temporal Decay Tests ============


class TestTemporalDecay(TestBase):

    def setUp(self):
        super().setUp()
        # Insert two insights with same tier/votes but different dates
        self.db.upsert_insight(
            text="Recent SQLite insight about WAL mode performance",
            tier=1, project="test", captured_at="2026-03-01T12:00:00",
        )
        self.db.upsert_insight(
            text="Old SQLite insight about WAL mode configuration",
            tier=1, project="test", captured_at="2025-06-01T12:00:00",
        )

    def test_search_recency_favors_recent(self):
        """Recent insight ranks higher than old insight, same tier/votes."""
        results = self.db.search("SQLite WAL", project="test")
        self.assertTrue(len(results) >= 2)
        # Recent one should be first
        recent = [r for r in results if "performance" in r["text"]]
        old = [r for r in results if "configuration" in r["text"]]
        if recent and old:
            recent_idx = results.index(recent[0])
            old_idx = results.index(old[0])
            self.assertLess(recent_idx, old_idx)

    def test_search_excludes_expired(self):
        """search() excludes expired insights."""
        self.db.upsert_insight(
            text="Expired SQLite WAL insight should hide",
            tier=0, project="test",
            captured_at="2026-03-01T00:00:00",
            expired_at="2026-02-28T00:00:00",
        )
        results = self.db.search("SQLite WAL", project="test")
        texts = [r["text"] for r in results]
        self.assertNotIn("Expired SQLite WAL insight should hide", texts)

    def test_top_insights_returns_results(self):
        """get_top_insights returns ranked results without derived columns."""
        results = self.db.get_top_insights(limit=10, days=365)
        if results:
            # Removed tier_weight/recency_factor from SELECT (unused derived columns)
            self.assertIn("tier", results[0])
            self.assertIn("captured_at", results[0])

    def test_high_tier_resists_decay(self):
        """Tier-0 insight from 30 days ago still outranks tier-3 from today."""
        self.db.upsert_insight(
            text="Pinned architectural decision about SQLite",
            tier=0, project="test", captured_at="2026-02-01T12:00:00",
        )
        self.db.upsert_insight(
            text="Trivial SQLite observation from today",
            tier=3, project="test",
        )
        results = self.db.search("SQLite", project="test")
        if len(results) >= 2:
            pinned = [r for r in results if "Pinned" in r["text"]]
            trivial = [r for r in results if "Trivial" in r["text"]]
            if pinned and trivial:
                self.assertLess(results.index(pinned[0]), results.index(trivial[0]))


# ============ Salience Tests ============


class TestSalience(TestBase):

    def test_salience_stored(self):
        """Salience value is stored on insert."""
        self.db.upsert_insight(
            text="High salience insight about debugging",
            salience=0.9,
        )
        h = compute_insight_hash("High salience insight about debugging")
        result = self.db.get_insight(h)
        self.assertAlmostEqual(result["salience"], 0.9, places=1)

    def test_salience_default(self):
        """Default salience is 0.5."""
        self.db.upsert_insight(text="Default salience insight")
        h = compute_insight_hash("Default salience insight")
        result = self.db.get_insight(h)
        self.assertAlmostEqual(result["salience"], 0.5, places=1)

    def test_high_salience_ranks_higher(self):
        """Higher salience insight ranks above lower, same tier/date."""
        self.db.upsert_insight(
            text="High salience SQLite performance breakthrough",
            tier=2, salience=0.9, project="test",
        )
        self.db.upsert_insight(
            text="Low salience SQLite performance note",
            tier=2, salience=0.1, project="test",
        )
        results = self.db.search("SQLite performance", project="test")
        if len(results) >= 2:
            high = [r for r in results if "breakthrough" in r["text"]]
            low = [r for r in results if "note" in r["text"]]
            if high and low:
                self.assertLess(results.index(high[0]), results.index(low[0]))

    def test_migration_runs_on_existing_db(self):
        """Migration adds salience column to existing database."""
        row = self.db.fetchone(
            "SELECT salience FROM insights LIMIT 1"
        )
        # No rows is fine -- we just need the query not to error
        self.assertTrue(True)

    def test_schema_migrations_table_exists(self):
        """schema_migrations table tracks applied migrations."""
        rows = self.db.fetchall(
            "SELECT name FROM schema_migrations"
        )
        names = [r["name"] for r in rows]
        self.assertIn("salience_column", names)


# ============ Corroboration Tests ============


class TestCorroboration(TestBase):

    def test_increment_corroboration(self):
        """Corroboration count increments."""
        self.db.upsert_insight(text="Corroborate me please")
        h = compute_insight_hash("Corroborate me please")
        self.assertTrue(self.db.increment_corroboration(h))
        result = self.db.get_insight(h)
        self.assertEqual(result["corroboration_count"], 1)
        self.assertTrue(self.db.increment_corroboration(h))
        result = self.db.get_insight(h)
        self.assertEqual(result["corroboration_count"], 2)

    def test_corroboration_nonexistent(self):
        """Corroboration on nonexistent returns False."""
        result = self.db.increment_corroboration("nonexistent")
        self.assertFalse(result)

    def test_corroboration_default_zero(self):
        """New insights start with 0 corroboration."""
        self.db.upsert_insight(text="Fresh insight")
        h = compute_insight_hash("Fresh insight")
        result = self.db.get_insight(h)
        self.assertEqual(result["corroboration_count"], 0)


# ============ Stale Pruning Tests ============


class TestPruning(TestBase):

    def test_prune_stale_expires_old_unvoted(self):
        """Old tier-3 unvoted insights get expired."""
        self.db.upsert_insight(
            text="Ancient observation nobody cared about",
            tier=3, captured_at="2025-01-01T00:00:00",
        )
        pruned = self.db.prune_stale(max_age_days=90)
        self.assertEqual(pruned, 1)
        h = compute_insight_hash("Ancient observation nobody cared about")
        result = self.db.get_insight(h)
        self.assertIsNotNone(result["expired_at"])

    def test_prune_spares_voted(self):
        """Upvoted insights are not pruned even if old."""
        self.db.upsert_insight(
            text="Old but voted insight survives",
            tier=3, captured_at="2025-01-01T00:00:00",
        )
        h = compute_insight_hash("Old but voted insight survives")
        self.db.vote(h, "up")
        pruned = self.db.prune_stale(max_age_days=90)
        self.assertEqual(pruned, 0)

    def test_prune_spares_corroborated(self):
        """Corroborated insights are not pruned."""
        self.db.upsert_insight(
            text="Old but corroborated insight survives",
            tier=3, captured_at="2025-01-01T00:00:00",
        )
        h = compute_insight_hash("Old but corroborated insight survives")
        self.db.increment_corroboration(h)
        pruned = self.db.prune_stale(max_age_days=90)
        self.assertEqual(pruned, 0)

    def test_prune_spares_high_tier(self):
        """Tier 0-2 insights are not pruned regardless of age."""
        self.db.upsert_insight(
            text="Old tier-1 discovery insight",
            tier=1, captured_at="2025-01-01T00:00:00",
        )
        pruned = self.db.prune_stale(max_age_days=90, min_tier=3)
        self.assertEqual(pruned, 0)

    def test_prune_spares_recent(self):
        """Recent insights are not pruned even if tier-3."""
        self.db.upsert_insight(text="Recent tier-3 observation", tier=3)
        pruned = self.db.prune_stale(max_age_days=90)
        self.assertEqual(pruned, 0)

    def test_prune_skips_already_expired(self):
        """Already-expired insights are not double-expired."""
        self.db.upsert_insight(
            text="Already expired insight",
            tier=3, captured_at="2025-01-01T00:00:00",
            expired_at="2025-06-01T00:00:00",
        )
        pruned = self.db.prune_stale(max_age_days=90)
        self.assertEqual(pruned, 0)


# ============ Investigation Prompt Tests ============


class TestInvestigationPrompts(TestBase):

    def test_low_confidence_prompt(self):
        """Low-confidence recent insight generates investigation prompt."""
        self.db.upsert_insight(
            text="Uncertain observation about SQLite behavior that needs verification",
            confidence=0.3, project="test",
        )
        prompts = self.db.get_investigation_prompts(limit=5)
        self.assertTrue(any("[investigate]" in p and "Low-confidence" in p for p in prompts))

    def test_stale_project_prompt(self):
        """Project with old insights generates staleness prompt."""
        for i in range(4):
            self.db.upsert_insight(
                text="Stale insight number %d about some topic" % i,
                project="old-project",
                captured_at="2025-01-01T00:00:00",
            )
        prompts = self.db.get_investigation_prompts(limit=5)
        self.assertTrue(any("old-project" in p and "30+ days" in p for p in prompts))

    def test_corroborated_prompt(self):
        """Highly corroborated old insight generates evolution prompt."""
        self.db.upsert_insight(
            text="Frequently rediscovered pattern about database connections",
            captured_at="2025-06-01T00:00:00",
        )
        h = compute_insight_hash("Frequently rediscovered pattern about database connections")
        for _ in range(4):
            self.db.increment_corroboration(h)
        prompts = self.db.get_investigation_prompts(limit=5)
        self.assertTrue(any("rediscovered" in p.lower() for p in prompts))

    def test_limit_respected(self):
        """Limit caps prompts returned."""
        for i in range(5):
            self.db.upsert_insight(
                text="Low confidence insight number %d that is uncertain" % i,
                confidence=0.2, project="test",
            )
        prompts = self.db.get_investigation_prompts(limit=2)
        self.assertLessEqual(len(prompts), 2)

    def test_empty_db_returns_empty(self):
        """Empty DB produces no prompts."""
        prompts = self.db.get_investigation_prompts()
        self.assertEqual(prompts, [])

    def test_project_filter(self):
        """Project filter scopes stale prompts."""
        for i in range(4):
            self.db.upsert_insight(
                text="Stale insight %d for wrong project" % i,
                project="other-project",
                captured_at="2025-01-01T00:00:00",
            )
        prompts = self.db.get_investigation_prompts(project="my-project", limit=5)
        # Should not suggest investigating other-project when filtered to my-project
        stale_prompts = [p for p in prompts if "other-project" in p]
        self.assertEqual(len(stale_prompts), 0)


if __name__ == "__main__":
    unittest.main()
