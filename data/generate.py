# data/generate.py

import json
import random
import os
from datetime import datetime, timedelta
from collections import Counter

COMPANIES = [
    {
        "variants": [
            "Sunrise Textile Manufacturing Co.",
            "Sunrise Textile Mfg Co",
            "SUNRISE TEXTILE MANUFACTURING",
            "Sunrise Textiles",
            "SR Textile Co",
        ],
        "country": "China",
        "industry": "Textiles",
    },
    {
        "variants": [
            "Golden Bridge Electronics Ltd.",
            "Golden Bridge Electronics",
            "GOLDEN BRIDGE ELEC LTD",
            "GBE Electronics",
            "Golden Bridge Tech",
        ],
        "country": "China",
        "industry": "Electronics",
    },
    {
        "variants": [
            "Meridian Seafood Processing Inc.",
            "Meridian Seafood",
            "MERIDIAN SEAFOOD PROCESSING",
            "Meridian Fish Processing Inc",
            "Meridian Seafood Corp",
        ],
        "country": "Thailand",
        "industry": "Food Processing",
    },
    {
        "variants": [
            "Atlas Garment Factory Ltd.",
            "Atlas Garments",
            "ATLAS GARMENT FACTORY LTD",
            "Atlas Clothing Factory",
            "Atlas Garment Co",
        ],
        "country": "Bangladesh",
        "industry": "Apparel",
    },
    {
        "variants": [
            "Pacific Rim Mining Corporation",
            "Pacific Rim Mining",
            "PACIFIC RIM MINING CORP",
            "Pacific Mining Corp",
            "PRM Corporation",
        ],
        "country": "Myanmar",
        "industry": "Mining",
    },
    {
        "variants": [
            "Evergreen Agricultural Exports Ltd.",
            "Evergreen Agriculture",
            "EVERGREEN AGRICULTURAL EXPORT",
            "Evergreen Agri Exports",
            "Evergreen Exports Ltd",
        ],
        "country": "Brazil",
        "industry": "Agriculture",
    },
    {
        "variants": [
            "Nova Chemical Industries Ltd.",
            "Nova Chemicals",
            "NOVA CHEMICAL IND",
            "Nova Chemical Co",
            "Nova Chem Industries Ltd",
        ],
        "country": "Iran",
        "industry": "Chemicals",
    },
    {
        "variants": [
            "Starlight Footwear Manufacturing Co.",
            "Starlight Footwear",
            "STARLIGHT FOOTWEAR MFG",
            "Starlight Shoes",
            "Starlight Manufacturing",
        ],
        "country": "Vietnam",
        "industry": "Footwear",
    },
    {
        "variants": [
            "Delta Port Logistics Ltd.",
            "Delta Logistics",
            "DELTA PORT LOGISTICS LTD",
            "Delta Port Services",
            "Delta Shipping Logistics",
        ],
        "country": "Russia",
        "industry": "Logistics",
    },
    {
        "variants": [
            "Horizon Tech Components Ltd.",
            "Horizon Tech",
            "HORIZON TECH COMPONENTS LTD",
            "Horizon Components",
            "Horizon Technology Co",
        ],
        "country": "North Korea",
        "industry": "Electronics",
    },
]

VIOLATIONS = {
    "forced_labor": {
        "severity": "Critical",
        "description_templates": [
            "Workers found confined to factory premises with movement restricted. Passports confiscated upon arrival from {count} workers.",
            "Subcontracted labor sourced from detention facilities. Workers unable to leave without financial penalty.",
            "Recruitment fees charged to {count} workers creating debt bondage conditions.",
            "Night shift workers locked in facility with no freedom to terminate employment.",
        ],
    },
    "child_labor": {
        "severity": "Critical",
        "description_templates": [
            "Workers aged 13-15 found operating heavy machinery across {count} production lines.",
            "School-age children employed during school hours in violation of minimum age requirements.",
            "Children under 16 found working night shifts exceeding 8 hours in hazardous conditions.",
        ],
    },
    "sanctions_violation": {
        "severity": "High",
        "description_templates": [
            "Entity listed on OFAC SDN list for transactions with sanctioned government entities.",
            "Shipments routed through sanctioned jurisdiction to circumvent export controls on {count} occasions.",
            "Financial transactions identified with sanctioned state-owned enterprise.",
            "Dual-use technology exports to sanctioned entity in violation of export control regulations.",
        ],
    },
    "environmental": {
        "severity": "Medium",
        "description_templates": [
            "Illegal discharge of industrial wastewater exceeding permitted levels by {count}x into local river.",
            "Unreported emissions of hazardous chemicals detected at facility boundary.",
            "Improper disposal of toxic waste adjacent to residential area affecting {count} households.",
        ],
    },
    "wage_theft": {
        "severity": "Medium",
        "description_templates": [
            "Workers paid below minimum wage across {count} pay periods. Total underpayment estimated at $2.3M.",
            "Overtime hours systematically uncompensated across {count} workers.",
            "Illegal deductions from wages for housing reducing effective pay below minimum wage.",
        ],
    },
}

JURISDICTIONS = [
    {
        "name": "USA",
        "agency": "US Customs and Border Protection",
        "list": "UFLPA Entity List",
    },
    {
        "name": "USA",
        "agency": "OFAC",
        "list": "SDN List",
    },
    {
        "name": "EU",
        "agency": "European Commission",
        "list": "EU Sanctions Map",
    },
    {
        "name": "UK",
        "agency": "HMRC Border Force",
        "list": "UK Global Sanctions List",
    },
    {
        "name": "Canada",
        "agency": "Canada Border Services Agency",
        "list": "Canadian Sanctions List",
    },
    {
        "name": "Australia",
        "agency": "Australian Border Force",
        "list": "Australian Sanctions List",
    },
]


def random_date(start_year=2018, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (
        start + timedelta(days=random.randint(0, delta.days))
    ).strftime("%Y-%m-%d")


def generate_records():
    records = []
    record_id = 1

    for company in COMPANIES:
        num_records = random.randint(3, 6)
        violation_types = random.sample(
            list(VIOLATIONS.keys()), min(3, len(VIOLATIONS))
        )

        for i in range(num_records):
            violation_type = violation_types[i % len(violation_types)]
            violation = VIOLATIONS[violation_type]
            jurisdiction = random.choice(JURISDICTIONS)

            # pick a random variant — this is the only name in the record
            # no canonical name stored anywhere in raw data
            name_at_border = random.choice(company["variants"])

            template = random.choice(violation["description_templates"])
            description = template.replace(
                "{count}", str(random.randint(50, 5000))
            )

            record = {
                "id": f"VR-{record_id:04d}",
                "name_at_border": name_at_border,
                "country_of_origin": company["country"],
                "industry": company["industry"],
                "violation_type": violation_type,
                "severity": violation["severity"],
                "jurisdiction": jurisdiction["name"],
                "reporting_agency": jurisdiction["agency"],
                "source_list": jurisdiction["list"],
                "date_reported": random_date(),
                "description": description,
                "status": random.choice(
                    ["Active", "Active", "Active", "Under Review", "Resolved"]
                ),
            }
            records.append(record)
            record_id += 1

    return records


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    random.seed(42)
    records = generate_records()

    output_path = "data/raw/violations.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {len(records)} violation records")
    print(f"Unique border names: {len(set(r['name_at_border'] for r in records))}")
    print(f"Saved to {output_path}")
    print()
    print("By severity:", dict(Counter(r["severity"] for r in records)))
    print("By violation type:", dict(Counter(r["violation_type"] for r in records)))
    print()

    # show sample — confirm no canonical name anywhere
    print("Sample record:")
    print(json.dumps(records[0], indent=2))