import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import random

_DATASET_DF = None

def _get_dataset():
    global _DATASET_DF
    if _DATASET_DF is None:
        try:
            print("[DEBUG] Loading clothing dataset from Kaggle...")
            _DATASET_DF = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "agrigorev/clothing-dataset-full",
                "images.csv",
            )
            # Filter out 'Skip', 'Not sure', 'Other'
            _DATASET_DF = _DATASET_DF[~_DATASET_DF['label'].isin(['Skip', 'Not sure', 'Other'])]
        except Exception as e:
            print(f"[WARNING] Failed to load Kaggle dataset: {e}")
            # Fallback to empty DF so we don't crash
            _DATASET_DF = pd.DataFrame(columns=['image', 'label', 'kids'])
    return _DATASET_DF

_COLOR_MAP = {
    "porcelain": [
        "icy blue", "silver", "soft lavender", "mint", "cool pink",
        "charcoal", "navy", "emerald green", "white", "black"
    ],
    "fair": [
        "peach", "soft pink", "baby blue", "mint green", "champagne",
        "light grey", "lavender", "coral", "ivory", "rose gold"
    ],
    "light": [
        "sage green", "dusty rose", "beige", "periwinkle", "camel",
        "olive", "mauve", "powder blue", "teal", "burgundy"
    ],
    "medium": [
        "olive", "terracotta", "forest green", "rust", "caramel",
        "deep teal", "mustard", "cinnamon", "amber", "navy"
    ],
    "tan": [
        "mustard", "rust", "coral", "deep teal", "warm brown",
        "burnt orange", "gold", "olive", "turquoise", "cream"
    ],
    "deep": [
        "royal blue", "emerald green", "gold", "burgundy", "plum",
        "magenta", "deep purple", "silver", "bright white", "yellow"
    ],
    "ebony": [
        "bright white", "cobalt blue", "fuchsia", "canary yellow", "tangerine",
        "gold", "silver", "cream", "electric lime", "crimson"
    ],
}

_GARMENT_MAP = {
    "male": [
        "kurta", "dress shirt", "sherwani", "shalwar kameez", "linen shirt",
        "polo shirt", "waistcoat", "blazer", "turtleneck", "bomber jacket",
    ],
    "female": [
        "kurti", "maxi dress", "abaya", "shalwar kameez", "lehenga",
        "salwar suit", "wrap dress", "anarkali", "embroidered blouse", "co-ord set",
    ],
    "unknown": [
        "kurta", "maxi dress", "linen shirt", "shalwar kameez",
        "wrap dress", "blazer", "anarkali", "polo shirt",
    ],
}

_STYLE_MAP = {
    "round": {
        "description": "Structured & elongating outfits",
        "tips": [
            "Choose deep V-necks or U-necks to visually lengthen the face.",
            "Opt for vertical stripes and narrow patterns.",
            "Wear fitted blazers and structured jackets with sharp shoulders.",
            "Avoid round or jewel necklines that mirror face shape.",
            "Tall collars and high necklines add vertical height.",
        ],
    },
    "long": {
        "description": "Layered & width-adding outfits",
        "tips": [
            "Horizontal patterns and wide prints add visual width.",
            "Choose boat necks, wide collars, and off-shoulder styles.",
            "Layer with chunky scarves, shawls, or wide dupattas.",
            "Go for cropped jackets — they break vertical length nicely.",
            "Bold, wide accessories (headbands, wide belts) balance proportions.",
        ],
    },
    "wide": {
        "description": "Slim-fit & elongating outfits",
        "tips": [
            "Monochrome outfits create a streamlined, elongated silhouette.",
            "Deep V-necks and narrow necklines draw the eye inward.",
            "Avoid extra-wide collars or very large hats.",
            "Slim-fit silhouettes in dark shades work best.",
            "Vertical embroidery or pintucks add slimming vertical lines.",
        ],
    },
    "oval": {
        "description": "Any style — oval is the most versatile face shape",
        "tips": [
            "Almost every neckline flatters an oval face — experiment freely.",
            "Classic tailored fits showcase your balanced proportions perfectly.",
            "Bold prints and statement patterns are great — you carry them well.",
            "Try asymmetric cuts and one-shoulder styles for a fashion-forward look.",
            "Strong accessories (chunky jewellery, wide belts) elevate any outfit.",
        ],
    },
}

_ACCESSORY_MAP = {
    "male": [
        "leather watch", "classic woven belt", "pocket square",
        "aviator sunglasses", "leather loafers", "cufflinks",
    ],
    "female": [
        "statement earrings", "silk dupatta", "embroidered clutch",
        "bangles / kadas", "strappy heels", "layered necklace",
    ],
    "unknown": [
        "wristwatch", "sunglasses", "leather bag",
        "scarf / dupatta", "classic belt",
    ],
}

_OCCASION_MAP = {
    "kurta": "casual / festive",
    "dress shirt": "formal / office",
    "sherwani": "wedding / formal",
    "shalwar kameez": "casual / traditional",
    "linen shirt": "casual / summer",
    "polo shirt": "smart casual",
    "waistcoat": "formal / layering",
    "blazer": "business / formal",
    "turtleneck": "smart casual / winter",
    "bomber jacket": "streetwear / casual",
    "kurti": "daily / casual",
    "maxi dress": "casual / evening",
    "abaya": "modest / formal",
    "lehenga": "wedding / festive",
    "salwar suit": "casual / traditional",
    "wrap dress": "casual / dinner",
    "anarkali": "festive / wedding",
    "embroidered blouse": "festive / formal",
    "co-ord set": "smart casual / trendy",
}


def get_recommendations(gender: str, skin_tone: str, face_shape: str) -> dict:
    g = gender.lower()
    df = _get_dataset()
    
    # 1. Determine base colors
    colors = _COLOR_MAP.get(skin_tone.lower(), ["navy", "white", "beige", "olive"])
    
    # 2. Get garments from dataset if available, otherwise fallback to map
    if not df.empty:
        # Filter for non-kids if possible (dataset has 'kids' column)
        adult_df = df[df['kids'] == False]
        if adult_df.empty: adult_df = df
        
        # Categorize labels into gender buckets (rough heuristic)
        male_labels = ['T-Shirt', 'Shirt', 'Pants', 'Outwear', 'Polo', 'Hoodie', 'Blazer']
        female_labels = ['T-Shirt', 'Shirt', 'Pants', 'Skirt', 'Top', 'Outwear', 'Dress', 'Blouse', 'Hoodie', 'Blazer']
        
        target_labels = male_labels if g == "male" else female_labels
        
        # Get matching items
        matching_items = adult_df[adult_df['label'].isin(target_labels)]
        if not matching_items.empty:
            # Pick a diverse set of 10 items
            sample_size = min(len(matching_items), 10)
            garments = matching_items.sample(sample_size)['label'].str.lower().unique().tolist()
        else:
            garments = _GARMENT_MAP.get(g, _GARMENT_MAP["unknown"])
    else:
        garments = _GARMENT_MAP.get(g, _GARMENT_MAP["unknown"])

    style_info = _STYLE_MAP.get(face_shape.lower(), _STYLE_MAP["oval"])
    accessories = _ACCESSORY_MAP.get(g, _ACCESSORY_MAP["unknown"])

    garments_detail = [
        {"name": item, "occasion": _OCCASION_MAP.get(item, "versatile")}
        for item in garments
    ]

    return {
        "colors": colors,
        "garments": garments,
        "garments_detail": garments_detail,
        "style": style_info["description"],
        "style_tips": style_info["tips"],
        "accessories": accessories,
    }
