from app.indexing.classifier import Category, classify_document, classify_query


def test_classifies_tax_document():
    assert classify_document("SRO 1213 of 2025 - Income Tax Ordinance Amendment") == Category.TAXATION


def test_classifies_monetary_policy_document():
    assert classify_document("SBP Monetary Policy Statement - Policy Rate Decision") == Category.MONETARY_POLICY


def test_classifies_capital_markets_query():
    assert classify_query("What are the PSX listing regulations for a new company?") == Category.CAPITAL_MARKETS


def test_unrecognized_text_falls_back_to_general():
    assert classify_document("Bank Holiday Notice") == Category.GENERAL