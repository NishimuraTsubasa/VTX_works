import pandas as pd

from stock_scoring_model.sector_grouping import apply_country_region_map, apply_sector_group_map, selected_interactions


def test_maps_and_interactions():
    data = pd.DataFrame({"country": ["US"], "sector": ["Financials"]})
    country_map = pd.DataFrame({"Country": ["US"], "Region": ["North_America"], "Enabled": [1]})
    sector_map = pd.DataFrame({"Sector": ["Financials"], "Sector_Group": ["Banks"], "Enabled": [1]})
    interactions = pd.DataFrame({"Sector_Group": ["Banks"], "Factor_Group": ["Value"], "Enabled": [1]})
    assert apply_country_region_map(data, "country", country_map).iloc[0] == "North_America"
    assert apply_sector_group_map(data, "sector", sector_map).iloc[0] == "Banks"
    assert ("Banks", "Value") in selected_interactions(interactions)
