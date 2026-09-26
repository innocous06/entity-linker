"""Synthetic benchmark dataset generator for multilingual entity resolution.

Generates mock business entity tables and ground truth labels with controlled
synthetic corruptions (letter doubling, transliteration, address variations,
distractors) to enable testing without proprietary competition data.
"""

import csv
import random
from pathlib import Path

# Seed for deterministic generation across runs
SEED = 42
random.seed(SEED)

SAMPLE_COMPANIES = [
    # India businesses
    ("Dream Construction", "16-11-23/37/A, Sagar Hotel Building, Osarambagh", "Hyderabad", "Telangana", "500036", "India", "Limited"),
    ("Green Logistics", "E-7, Second Floor, South Extension", "New Delhi", "Delhi", "110049", "India", "Private Limited"),
    ("Silver Constructions", "101 CTS No, 917/18 F P No, British Library Road", "Pune", "Maharashtra", "411004", "India", "LLP"),
    ("Shiva Management", "D No. 20-6-3/12, Dantuluri Vari Street, Ayodyanagar", "Vijayawada", "Andhra Pradesh", "520003", "India", "Private Limited"),
    ("Alpha Media", "28-B, Chand Bihari Nagar, Khatipura", "Jaipur", "Rajasthan", "302012", "India", "Private Limited"),
    ("Surya Energy", "C-252-A, Malviya Nagar, Hans Marg", "Jaipur", "Rajasthan", "302017", "India", "Private Limited"),
    ("Indian Brothers", "17/1, Lower Ground Floor, 1st Cross, Lalbagh Road", "Bangalore", "Karnataka", "560027", "India", "Private Limited"),
    ("Apex Technologies", "Plot 44, Electronic City Phase 1, Hosur Road", "Bangalore", "Karnataka", "560100", "India", "LLP"),
    ("Kalyan Jewellers", "Shop 12, Ground Floor, MG Road, Somajiguda", "Hyderabad", "Telangana", "500082", "India", "Limited"),
    ("Maruthi Traders", "4-1-898, Tilak Road, Abids", "Hyderabad", "Telangana", "500001", "India", "Private Limited"),
    ("Creative Solutions", "SCO 88-89, Sector 17-C", "Chandigarh", "Punjab", "160017", "India", "Private Limited"),
    ("Modern Engineering", "B-14, Industrial Area, Phase II, Mayapuri", "New Delhi", "Delhi", "110064", "India", "Limited"),
    ("Royal Healthcare", "23/4, Park Street, Near Park Plaza", "Kolkata", "West Bengal", "700016", "India", "Private Limited"),
    ("Prime Retail", "55, Commercial Street, Tasker Town", "Bangalore", "Karnataka", "560001", "India", "LLP"),
    ("Universal Exports", "Office 402, Trade Centre, BKC, Bandra", "Mumbai", "Maharashtra", "400051", "India", "Limited"),

    # US businesses
    ("Pacific Coast Logistics", "1420 Harbor Blvd, Suite 300", "Long Beach", "CA", "90802", "US", "LLC"),
    ("Hudson River Consulting", "350 5th Avenue, 59th Floor", "New York", "NY", "10118", "US", "Inc"),
    ("Lone Star Manufacturing", "1200 Industrial Parkway, Building B", "Houston", "TX", "77041", "US", "Corp"),
    ("Midwest Agricultural Supplies", "4450 Cornhusker Highway", "Lincoln", "NE", "68504", "US", "LLC"),
    ("Summit Peak Software", "1801 California Street, Suite 2400", "Denver", "CO", "80202", "US", "Inc"),
    ("Great Lakes Distribution", "600 Renaissance Center, Suite 1400", "Detroit", "MI", "48243", "US", "Corporation"),
    ("Chesapeake Maritime Services", "201 E Baltimore Street, 12th Floor", "Baltimore", "MD", "21202", "US", "LLC"),
    ("Sonoran Desert Energy", "2400 E Camelback Road, Suite 700", "Phoenix", "AZ", "85016", "US", "Corp"),
    ("Cascade Timber Holdings", "111 SW 5th Avenue, Suite 1900", "Portland", "OR", "97204", "US", "LLC"),
    ("New England Bioscience", "75 Cambridge Parkway, Suite 400", "Cambridge", "MA", "02142", "US", "Inc"),

    # France businesses
    ("Atelier Du Rhone", "14 Rue De La Republique", "Lyon", "Auvergne-Rhone-Alpes", "69002", "France", "SARL"),
    ("Bordeaux Vignobles Distribution", "25 Cours De L Intendance", "Bordeaux", "Nouvelle-Aquitaine", "33000", "France", "SAS"),
    ("Parisienne De Conseils", "42 Avenue Montaigne", "Paris", "Ile-De-France", "75008", "France", "Societe"),
    ("Provence Logistique Transport", "120 Boulevard National", "Marseille", "Provence-Alpes-Cote d Azur", "13003", "France", "SARL"),
    ("Nord Industrie Mecanique", "58 Rue Nationale", "Lille", "Hauts-De-France", "59000", "France", "SAS"),
]

# Distractor noise entities (never match S1)
DISTRACTORS = [
    ("Zenith Global Holdings", "100 Broadway, 18th Floor", "New York", "NY", "10005", "US", "LLC"),
    ("Evergreen Retail Partners", "88 King Street, Suite 200", "Seattle", "WA", "98104", "US", "Inc"),
    ("Blue Horizon Maritime", "700 S Biscayne Blvd", "Miami", "FL", "33131", "US", "Corp"),
    ("Om Sai Electronics", "22-1-45, Sultan Bazar", "Hyderabad", "Telangana", "500095", "India", "Private Limited"),
    ("Balaji Auto Spares", "Shop 4, Gandhi Road", "Vijayawada", "Andhra Pradesh", "520001", "India", "LLP"),
    ("Comptoir De L Atlantique", "8 Quai De La Fosse", "Nantes", "Pays de la Loire", "44000", "France", "SARL"),
]

def corrupt_name(name, legal_suffix, country):
    """Applies synthetic corruptions mirroring competition benchmark noise."""
    corrupted = name

    # Vowel/consonant doubling or phonetic shift
    replacements = [
        ("Dream", "ddriim"),
        ("Green", "griin"),
        ("Silver", "silvr"),
        ("Shiva", "shiv"),
        ("Construction", "knsttrkssn"),
        ("Constructions", "kNsttrkshNs"),
        ("Logistics", "loNjisttiks"),
        ("Energy", "enrjii"),
        ("Media", "miiddiyaa"),
        ("Brothers", "Brothers Private"),
        ("Pacific", "Pasifik"),
        ("Software", "Softwre"),
        ("Agricultural", "Agri"),
        ("Manufacturing", "Mfg"),
        ("Consulting", "Cnslting"),
        ("Distribution", "Distrib"),
        ("Atelier", "Attelier"),
        ("Logistique", "Logistik"),
        ("Mecanique", "Mekanik"),
    ]
    for orig, rep in replacements:
        if orig in corrupted and random.random() < 0.7:
            corrupted = corrupted.replace(orig, rep)

    # Suffix transliteration or abbreviation
    suffix_map = {
        "Private Limited": ["praaivett limittedd", "praiveett limittedd", "Pvt Ltd", "P. Ltd."],
        "Limited": ["limittedd", "limirrrrdd", "Ltd", "Ltd."],
        "LLP": ["elelpii", "elelpi", "L.L.P."],
        "LLC": ["L.L.C.", "LLC Corp"],
        "Inc": ["Inc.", "Incorporated"],
        "Corp": ["Corporation", "Corp."],
        "SARL": ["S.A.R.L.", "Societe A Responsabilite Limitee"],
        "SAS": ["S.A.S.", "Societe Par Actions Simplifiee"],
    }
    sfx_choices = suffix_map.get(legal_suffix, [legal_suffix])
    chosen_sfx = random.choice(sfx_choices)

    # Optional prefix injection (M/s, Dr, domain suffix)
    prefix = ""
    suffix_tail = ""
    r = random.random()
    if r < 0.15:
        prefix = "M/s "
    elif r < 0.25 and country == "India":
        prefix = "Dr "
    elif r < 0.35:
        suffix_tail = ".com"

    return f"{prefix}{corrupted} {chosen_sfx}{suffix_tail}".strip()

def corrupt_address(street, city, state, pincode):
    """Applies realistic address token reordering and abbreviation."""
    st = street
    if "Road" in st and random.random() < 0.5:
        st = st.replace("Road", "Rd")
    if "Street" in st and random.random() < 0.5:
        st = st.replace("Street", "St")
    if "Floor" in st and random.random() < 0.5:
        st = st.replace("Floor", "Flr")
    if "Avenue" in st and random.random() < 0.5:
        st = st.replace("Avenue", "Ave")
    if "Suite" in st and random.random() < 0.5:
        st = st.replace("Suite", "Ste")

    parts = [st, city, state]
    if random.random() < 0.3:
        # Invert order of street and city
        parts = [city, st, state]

    if pincode and random.random() < 0.6:
        parts.append(pincode)

    return ", ".join(parts)

def generate_mock_datasets(output_dir: Path):
    """Generates train and test TSV files according to competition specifications."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Split
    train_s1 = []
    train_s2 = []
    train_s3 = []
    ground_truth = []

    s1_counter = 1
    s2_counter = 1
    s3_counter = 1

    BRANCHES = ["", " North", " South", " East", " West"]
    for branch_idx, branch in enumerate(BRANCHES[:3]):
        for name, street, city, state, pincode, country, legal_sfx in SAMPLE_COMPANIES:
            s1_id = f"S1-{s1_counter:06d}"
            s1_counter += 1

            b_name = f"{name}{branch}"
            s1_name = f"{b_name} {legal_sfx}".strip()
            s1_addr = f"{street}, {city}, {state} {pincode}".strip()
            train_s1.append((s1_id, s1_name, s1_addr, country))

            matched_ids = []

            # Target S2 match (probability 0.90)
            if random.random() < 0.90:
                s2_id = f"S2-{s2_counter:06d}"
                s2_counter += 1
                s2_name = corrupt_name(b_name, legal_sfx, country)
                s2_addr = corrupt_address(street, city, state, pincode)
                train_s2.append((s2_id, s2_name, s2_addr, country))
                matched_ids.append(s2_id)

            # Target S3 match (probability 0.85)
            if random.random() < 0.85:
                s3_id = f"S3-{s3_counter:06d}"
                s3_counter += 1
                s3_name = corrupt_name(b_name, legal_sfx, country)
                s3_addr = corrupt_address(street, city, state, pincode)
                train_s3.append((s3_id, s3_name, s3_addr, country))
                matched_ids.append(s3_id)

            ground_truth.append((s1_id, ",".join(matched_ids)))

    # Add distractors to S2 and S3
    for name, street, city, state, pincode, country, legal_sfx in DISTRACTORS:
        s2_id = f"S2-{s2_counter:06d}"
        s2_counter += 1
        train_s2.append((s2_id, f"{name} {legal_sfx}", f"{street}, {city}, {state} {pincode}", country))

        s3_id = f"S3-{s3_counter:06d}"
        s3_counter += 1
        train_s3.append((s3_id, f"{name} {legal_sfx}", f"{street}, {city}, {state} {pincode}", country))

    # Shuffle targets to test position-independent indexing
    random.shuffle(train_s2)
    random.shuffle(train_s3)

    # 2. Test Split (half of companies + distractors, France included)
    test_s1 = []
    test_s2 = []
    test_s3 = []
    for name, street, city, state, pincode, country, legal_sfx in SAMPLE_COMPANIES[::2]:
        t1_id = f"S1-{s1_counter:06d}"
        s1_counter += 1
        test_s1.append((t1_id, f"{name} {legal_sfx}", f"{street}, {city}, {state} {pincode}", country))

        t2_id = f"S2-{s2_counter:06d}"
        s2_counter += 1
        test_s2.append((t2_id, corrupt_name(name, legal_sfx, country), corrupt_address(street, city, state, pincode), country))

        t3_id = f"S3-{s3_counter:06d}"
        s3_counter += 1
        test_s3.append((t3_id, corrupt_name(name, legal_sfx, country), corrupt_address(street, city, state, pincode), country))

    for name, street, city, state, pincode, country, legal_sfx in DISTRACTORS[:3]:
        t2_id = f"S2-{s2_counter:06d}"
        s2_counter += 1
        test_s2.append((t2_id, f"{name} {legal_sfx}", f"{street}, {city}, {state}", country))

    random.shuffle(test_s2)
    random.shuffle(test_s3)

    def write_tsv(path, rows, headers):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(headers)
            w.writerows(rows)

    src_headers = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(output_dir / "train_source1.tsv", train_s1, src_headers)
    write_tsv(output_dir / "train_source2.tsv", train_s2, src_headers)
    write_tsv(output_dir / "train_source3.tsv", train_s3, src_headers)
    write_tsv(output_dir / "train_ground_truth.tsv", ground_truth, ["entity_id", "matched_entities"])

    write_tsv(output_dir / "test_source1.tsv", test_s1, src_headers)
    write_tsv(output_dir / "test_source2.tsv", test_s2, src_headers)
    write_tsv(output_dir / "test_source3.tsv", test_s3, src_headers)

    print(f"[OK] Generated synthetic dataset in: {output_dir}")
    print(f"  Train: S1={len(train_s1)}, S2={len(train_s2)}, S3={len(train_s3)}, GT={len(ground_truth)}")
    print(f"  Test:  S1={len(test_s1)}, S2={len(test_s2)}, S3={len(test_s3)}")

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "sample_data"
    generate_mock_datasets(out)
