from biome.text import text_cleaning


def test_make_rule_callable():
    clean_text = text_cleaning.strip_spaces("   This is a text \n\n")
    assert clean_text == "This is a text"
